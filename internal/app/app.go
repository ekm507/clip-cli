package app

import (
	"errors"
	"flag"
	"fmt"
	"sort"

	"clip_cli/internal/clip"
	"clip_cli/internal/config"
	"clip_cli/internal/storage"
	"clip_cli/internal/vector"
)

type Application struct {
	cfg config.Config
}

type SearchResult struct {
	Path  string
	Score float32
}

func New(cfg config.Config) Application {
	return Application{cfg: cfg}
}

func (a Application) Run(args []string) error {
	cleanup, err := clip.InitializeEnvironment(a.cfg.ORTSharedLibPath)
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	defer cleanup()

	store, err := storage.Open(a.cfg.DatabasePath)
	if err != nil {
		return fmt.Errorf("failed to open database: %w", err)
	}
	defer store.Close()

	addCmd := flag.NewFlagSet("add", flag.ContinueOnError)
	addImagePath := addCmd.String("image", "", "Path to the image file")

	searchCmd := flag.NewFlagSet("search", flag.ContinueOnError)
	searchText := searchCmd.String("text", "", "Text query to search for")
	searchLimit := searchCmd.Int("limit", 5, "Number of results to show")

	if len(args) < 1 {
		return errors.New("usage: clip-cli [add|search] [options]")
	}

	runner := clip.NewRunner(a.cfg)

	switch args[0] {
	case "add":
		if err := addCmd.Parse(args[1:]); err != nil {
			return err
		}
		if *addImagePath == "" {
			return errors.New("image path is required: --image path/to/img.jpg")
		}
		return a.handleAdd(store, runner, *addImagePath)
	case "search":
		if err := searchCmd.Parse(args[1:]); err != nil {
			return err
		}
		if *searchText == "" {
			return errors.New("search text is required: --text \"your query\"")
		}
		return a.handleSearch(store, runner, *searchText, *searchLimit)
	default:
		return fmt.Errorf("invalid command %q. use add or search", args[0])
	}
}

func (a Application) handleAdd(store *storage.Store, runner clip.Runner, imagePath string) error {
	fmt.Println("Processing image...")
	pixelValues, err := clip.ProcessImageExact(imagePath)
	if err != nil {
		return fmt.Errorf("failed to process image: %w", err)
	}

	fmt.Println("Extracting image embedding...")
	embedding, err := runner.RunVisionModel(pixelValues)
	if err != nil {
		return fmt.Errorf("failed to run vision model: %w", err)
	}

	vector.L2Normalize(embedding)
	if err := store.UpsertImage(imagePath, embedding); err != nil {
		return fmt.Errorf("failed to save image embedding: %w", err)
	}

	fmt.Printf("Image %s added successfully.\n", imagePath)
	return nil
}

func (a Application) handleSearch(store *storage.Store, runner clip.Runner, query string, limit int) error {
	fmt.Println("Tokenizing text...")
	ids, masks, err := clip.ProcessText(query, a.cfg.TokenizerPath, a.cfg.MaxTokens)
	if err != nil {
		return fmt.Errorf("failed to tokenize query text: %w", err)
	}

	fmt.Println("Extracting query embedding...")
	queryEmbedding, err := runner.RunTextModel(ids, masks)
	if err != nil {
		return fmt.Errorf("failed to run text model: %w", err)
	}
	vector.L2Normalize(queryEmbedding)

	records, err := store.ListImages()
	if err != nil {
		return fmt.Errorf("failed to read image embeddings: %w", err)
	}

	results := make([]SearchResult, 0, len(records))
	for _, record := range records {
		score := vector.DotProduct(queryEmbedding, record.Embedding)
		results = append(results, SearchResult{Path: record.Path, Score: score})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	fmt.Printf("\n--- Search results for: %q ---\n", query)
	for i := 0; i < limit && i < len(results); i++ {
		fmt.Printf("%d. similarity: %.4f | path: %s\n", i+1, results[i].Score, results[i].Path)
	}

	return nil
}
