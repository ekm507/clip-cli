package app

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
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

const (
	embedFormatJSON   = "json"
	embedFormatBase64 = "base64"
	embedFormatF32LE  = "f32le"
)

type fileOutput struct {
	Files []string `json:"files"`
}

type embeddingOutput struct {
	Embedding []float32 `json:"embedding"`
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

	var store *storage.Store
	defer func() {
		if store != nil {
			_ = store.Close()
		}
	}()

	openStore := func() (*storage.Store, error) {
		if store != nil {
			return store, nil
		}
		dbStore, err := storage.Open(a.cfg.DatabasePath, a.cfg.EmbeddingDim)
		if err != nil {
			return nil, fmt.Errorf("failed to open database: %w", err)
		}
		store = dbStore
		return store, nil
	}

	addCmd := flag.NewFlagSet("add", flag.ContinueOnError)
	addImagePath := addCmd.String("image", "", "Path to the image file")

	searchCmd := flag.NewFlagSet("search", flag.ContinueOnError)
	searchText := searchCmd.String("text", "", "Text query to search for")
	searchLimit := searchCmd.Int("limit", 5, "Number of results to show")

	searchImageCmd := flag.NewFlagSet("search-image", flag.ContinueOnError)
	searchImagePath := searchImageCmd.String("image", "", "Path to the query image file")
	searchImageLimit := searchImageCmd.Int("limit", 5, "Number of results to show")

	embedCmd := flag.NewFlagSet("embed", flag.ContinueOnError)
	embedText := embedCmd.String("text", "", "Text input to embed")
	embedImage := embedCmd.String("image", "", "Image input to embed")
	embedFormat := embedCmd.String("format", embedFormatJSON, "Output format: json|base64|f32le")
	embedOut := embedCmd.String("out", "", "Optional output file path")

	if len(args) < 1 {
		return errors.New("usage: clip-cli [add|search|search-image|embed] [options]")
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
		store, err := openStore()
		if err != nil {
			return err
		}
		return a.handleAdd(store, runner, *addImagePath, jsonOutput)
	case "search":
		if err := searchCmd.Parse(args[1:]); err != nil {
			return err
		}
		if *searchText == "" {
			return errors.New("search text is required: --text \"your query\"")
		}
		store, err := openStore()
		if err != nil {
			return err
		}
		return a.handleSearch(store, runner, *searchText, *searchLimit, jsonOutput)
	case "search-image":
		if err := searchImageCmd.Parse(args[1:]); err != nil {
			return err
		}
		if *searchImagePath == "" {
			return errors.New("search image path is required: --image path/to/img.jpg")
		}
		store, err := openStore()
		if err != nil {
			return err
		}
		return a.handleSearchImage(store, runner, *searchImagePath, *searchImageLimit, jsonOutput)
	case "embed":
		if err := embedCmd.Parse(args[1:]); err != nil {
			return err
		}
		if (*embedText == "" && *embedImage == "") || (*embedText != "" && *embedImage != "") {
			return errors.New("exactly one of --text or --image is required for embed")
		}
		format := strings.ToLower(*embedFormat)
		if jsonOutput && format != embedFormatJSON {
			return errors.New("--json cannot be combined with non-json --format")
		}
		return a.handleEmbed(runner, *embedText, *embedImage, format, *embedOut)
	default:
		return fmt.Errorf("invalid command %q. use add, search, search-image, or embed", args[0])
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

func (a Application) handleSearchImage(store *storage.Store, runner clip.Runner, imagePath string, limit int, jsonOutput bool) error {
	fmt.Fprintln(os.Stderr, "Processing query image...")
	pixelValues, err := clip.ProcessImageExact(imagePath)
	if err != nil {
		return fmt.Errorf("failed to process query image: %w", err)
	}

	fmt.Fprintln(os.Stderr, "Extracting query image embedding...")
	queryEmbedding, err := runner.RunVisionModel(pixelValues)
	if err != nil {
		return fmt.Errorf("failed to run vision model for query image: %w", err)
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

func (a Application) handleEmbed(runner clip.Runner, text string, imagePath string, format string, outPath string) error {
	var embedding []float32
	var err error

	if text != "" {
		fmt.Fprintln(os.Stderr, "Tokenizing text...")
		ids, masks, textErr := clip.ProcessText(text, a.cfg.TokenizerPath, a.cfg.MaxTokens)
		if textErr != nil {
			return fmt.Errorf("failed to tokenize text: %w", textErr)
		}

		fmt.Fprintln(os.Stderr, "Extracting text embedding...")
		embedding, err = runner.RunTextModel(ids, masks)
		if err != nil {
			return fmt.Errorf("failed to run text model: %w", err)
		}
	} else {
		fmt.Fprintln(os.Stderr, "Processing image...")
		pixelValues, imageErr := clip.ProcessImageExact(imagePath)
		if imageErr != nil {
			return fmt.Errorf("failed to process image: %w", imageErr)
		}

		fmt.Fprintln(os.Stderr, "Extracting image embedding...")
		embedding, err = runner.RunVisionModel(pixelValues)
		if err != nil {
			return fmt.Errorf("failed to run vision model: %w", err)
		}
	}

	vector.L2Normalize(embedding)

	data, binaryOutput, err := formatEmbedding(embedding, format)
	if err != nil {
		return err
	}

	if outPath != "" {
		if err := os.WriteFile(outPath, data, 0o644); err != nil {
			return fmt.Errorf("failed to write output file: %w", err)
		}
		fmt.Fprintln(os.Stdout, outPath)
		return nil
	}

	if binaryOutput {
		_, err := os.Stdout.Write(data)
		return err
	}

	_, err = os.Stdout.Write(data)
	return err
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

func formatEmbedding(embedding []float32, format string) ([]byte, bool, error) {
	switch format {
	case embedFormatJSON:
		data, err := json.Marshal(embeddingOutput{Embedding: embedding})
		if err != nil {
			return nil, false, fmt.Errorf("failed to marshal embedding json: %w", err)
		}
		return append(data, '\n'), false, nil
	case embedFormatBase64:
		payload := base64.StdEncoding.EncodeToString(float32ToBytes(embedding))
		return []byte(payload + "\n"), false, nil
	case embedFormatF32LE:
		return float32ToBytes(embedding), true, nil
	default:
		return nil, false, fmt.Errorf("invalid embed format %q (supported: %s, %s, %s)", format, embedFormatJSON, embedFormatBase64, embedFormatF32LE)
	}
}

func float32ToBytes(values []float32) []byte {
	buf := new(bytes.Buffer)
	_ = binary.Write(buf, binary.LittleEndian, values)
	return buf.Bytes()
}
