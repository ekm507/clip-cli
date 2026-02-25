package app

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"strings"

	"clip_cli/internal/clip"
	"clip_cli/internal/config"
	"clip_cli/internal/storage"
	"clip_cli/internal/vector"
)

type Application struct {
	cfg config.Config
}

type fileOutput struct {
	Files []string `json:"files"`
}

func New(cfg config.Config) Application {
	return Application{cfg: cfg}
}

func (a Application) Run(args []string) error {
	args, jsonOutput := extractJSONFlag(args)

	cleanup, err := clip.InitializeEnvironment(a.cfg.ORTSharedLibPath)
	if err != nil {
		return fmt.Errorf("failed to initialize ONNX Runtime: %w", err)
	}
	defer cleanup()

	store, err := storage.Open(a.cfg.DatabasePath, a.cfg.EmbeddingDim)
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
		return a.handleAdd(store, runner, *addImagePath, jsonOutput)
	case "search":
		if err := searchCmd.Parse(args[1:]); err != nil {
			return err
		}
		if *searchText == "" {
			return errors.New("search text is required: --text \"your query\"")
		}
		return a.handleSearch(store, runner, *searchText, *searchLimit, jsonOutput)
	default:
		return fmt.Errorf("invalid command %q. use add or search", args[0])
	}
}

func (a Application) handleAdd(store *storage.Store, runner clip.Runner, imagePath string, jsonOutput bool) error {
	fmt.Fprintln(os.Stderr, "Processing image...")
	pixelValues, err := clip.ProcessImageExact(imagePath)
	if err != nil {
		return fmt.Errorf("failed to process image: %w", err)
	}

	fmt.Fprintln(os.Stderr, "Extracting image embedding...")
	embedding, err := runner.RunVisionModel(pixelValues)
	if err != nil {
		return fmt.Errorf("failed to run vision model: %w", err)
	}

	vector.L2Normalize(embedding)
	if err := store.UpsertImage(imagePath, embedding); err != nil {
		return fmt.Errorf("failed to save image embedding: %w", err)
	}

	return emitFiles([]string{imagePath}, jsonOutput)
}

func (a Application) handleSearch(store *storage.Store, runner clip.Runner, query string, limit int, jsonOutput bool) error {
	fmt.Fprintln(os.Stderr, "Tokenizing text...")
	ids, masks, err := clip.ProcessText(query, a.cfg.TokenizerPath, a.cfg.MaxTokens)
	if err != nil {
		return fmt.Errorf("failed to tokenize query text: %w", err)
	}

	fmt.Fprintln(os.Stderr, "Extracting query embedding...")
	queryEmbedding, err := runner.RunTextModel(ids, masks)
	if err != nil {
		return fmt.Errorf("failed to run text model: %w", err)
	}
	vector.L2Normalize(queryEmbedding)

	results, err := store.SearchByEmbedding(queryEmbedding, limit)
	if err != nil {
		return fmt.Errorf("failed to search image embeddings: %w", err)
	}

	paths := make([]string, 0, len(results))
	for _, result := range results {
		paths = append(paths, result.Path)
	}

	return emitFiles(paths, jsonOutput)
}

func emitFiles(files []string, jsonOutput bool) error {
	if jsonOutput {
		return json.NewEncoder(os.Stdout).Encode(fileOutput{Files: files})
	}
	for _, file := range files {
		fmt.Fprintln(os.Stdout, file)
	}
	return nil
}

func extractJSONFlag(args []string) ([]string, bool) {
	filtered := make([]string, 0, len(args))
	jsonOutput := false

	for _, arg := range args {
		switch {
		case arg == "--json":
			jsonOutput = true
		case strings.HasPrefix(arg, "--json="):
			value := strings.TrimPrefix(arg, "--json=")
			jsonOutput = value == "1" || strings.EqualFold(value, "true")
		default:
			filtered = append(filtered, arg)
		}
	}

	return filtered, jsonOutput
}
