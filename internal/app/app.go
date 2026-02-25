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
	if len(args) == 0 || args[0] == "-h" || args[0] == "--help" || args[0] == "help" {
		printRootHelp()
		return nil
	}

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
	addCmd.SetOutput(os.Stdout)
	var addImages stringSliceFlag
	addCmd.Var(&addImages, "image", "Image path to add (repeatable)")
	addCmd.Usage = func() {
		fmt.Fprintln(os.Stdout, "Usage: clip-cli add --image <path> [--image <path> ...] [path ...] [--json]")
		addCmd.PrintDefaults()
	}

	searchCmd := flag.NewFlagSet("search", flag.ContinueOnError)
	searchCmd.SetOutput(os.Stdout)
	searchText := searchCmd.String("text", "", "Text query to search for")
	searchImage := searchCmd.String("image", "", "Image query path")
	searchLimit := searchCmd.Int("limit", 5, "Number of results to show")
	searchCmd.Usage = func() {
		fmt.Fprintln(os.Stdout, "Usage: clip-cli search (--text <query> | --image <path>) [--limit N] [--json]")
		searchCmd.PrintDefaults()
	}

	embedCmd := flag.NewFlagSet("embed", flag.ContinueOnError)
	embedCmd.SetOutput(os.Stdout)
	embedText := embedCmd.String("text", "", "Text input to embed")
	embedImage := embedCmd.String("image", "", "Image input to embed")
	embedFormat := embedCmd.String("format", embedFormatJSON, "Output format: json|base64|f32le")
	embedOut := embedCmd.String("out", "", "Optional output file path")
	embedCmd.Usage = func() {
		fmt.Fprintln(os.Stdout, "Usage: clip-cli embed (--text <query> | --image <path>) [--format json|base64|f32le] [--out file]")
		embedCmd.PrintDefaults()
	}

	runner := clip.NewRunner(a.cfg)

	switch args[0] {
	case "add":
		if err := addCmd.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		images := make([]string, 0, len(addImages)+len(addCmd.Args()))
		images = append(images, addImages...)
		images = append(images, addCmd.Args()...)
		if len(images) == 0 {
			return errors.New("at least one image path is required")
		}
		store, err := openStore()
		if err != nil {
			return err
		}
		return a.handleAddMany(store, runner, images, jsonOutput)
	case "search":
		if err := searchCmd.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
			return err
		}
		if (*searchText == "" && *searchImage == "") || (*searchText != "" && *searchImage != "") {
			return errors.New("exactly one of --text or --image is required for search")
		}
		store, err := openStore()
		if err != nil {
			return err
		}
		return a.handleSearch(store, runner, *searchText, *searchImage, *searchLimit, jsonOutput)
	case "embed":
		if err := embedCmd.Parse(args[1:]); err != nil {
			if errors.Is(err, flag.ErrHelp) {
				return nil
			}
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
		return fmt.Errorf("invalid command %q. use add, search, or embed", args[0])
	}
}

func (a Application) handleAddMany(store *storage.Store, runner clip.Runner, imagePaths []string, jsonOutput bool) error {
	added := make([]string, 0, len(imagePaths))
	for _, imagePath := range imagePaths {
		fmt.Fprintln(os.Stderr, "Processing image...", imagePath)
		pixelValues, err := clip.ProcessImageExact(imagePath)
		if err != nil {
			return fmt.Errorf("failed to process image %q: %w", imagePath, err)
		}

		fmt.Fprintln(os.Stderr, "Extracting image embedding...", imagePath)
		embedding, err := runner.RunVisionModel(pixelValues)
		if err != nil {
			return fmt.Errorf("failed to run vision model for %q: %w", imagePath, err)
		}

		vector.L2Normalize(embedding)
		if err := store.UpsertImage(imagePath, embedding); err != nil {
			return fmt.Errorf("failed to save image embedding for %q: %w", imagePath, err)
		}
		added = append(added, imagePath)
	}

	return emitFiles(added, jsonOutput)
}

func (a Application) handleSearch(store *storage.Store, runner clip.Runner, textQuery string, imageQuery string, limit int, jsonOutput bool) error {
	var queryEmbedding []float32
	var err error

	if textQuery != "" {
		fmt.Fprintln(os.Stderr, "Tokenizing text...")
		ids, masks, textErr := clip.ProcessText(textQuery, a.cfg.TokenizerPath, a.cfg.MaxTokens)
		if textErr != nil {
			return fmt.Errorf("failed to tokenize query text: %w", textErr)
		}

		fmt.Fprintln(os.Stderr, "Extracting query embedding...")
		queryEmbedding, err = runner.RunTextModel(ids, masks)
		if err != nil {
			return fmt.Errorf("failed to run text model: %w", err)
		}
	} else {
		fmt.Fprintln(os.Stderr, "Processing query image...")
		pixelValues, imageErr := clip.ProcessImageExact(imageQuery)
		if imageErr != nil {
			return fmt.Errorf("failed to process query image: %w", imageErr)
		}

		fmt.Fprintln(os.Stderr, "Extracting query image embedding...")
		queryEmbedding, err = runner.RunVisionModel(pixelValues)
		if err != nil {
			return fmt.Errorf("failed to run vision model for query image: %w", err)
		}
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

type stringSliceFlag []string

func (s *stringSliceFlag) String() string {
	return strings.Join(*s, ",")
}

func (s *stringSliceFlag) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func printRootHelp() {
	fmt.Fprintln(os.Stdout, "Usage: clip-cli <command> [options]")
	fmt.Fprintln(os.Stdout, "")
	fmt.Fprintln(os.Stdout, "Commands:")
	fmt.Fprintln(os.Stdout, "  add     Add one or more images to the vector index")
	fmt.Fprintln(os.Stdout, "  search  Search by text or image")
	fmt.Fprintln(os.Stdout, "  embed   Output embedding for text or image")
	fmt.Fprintln(os.Stdout, "")
	fmt.Fprintln(os.Stdout, "Global flags:")
	fmt.Fprintln(os.Stdout, "  --json  Output machine-readable JSON where supported")
	fmt.Fprintln(os.Stdout, "")
	fmt.Fprintln(os.Stdout, "Run 'clip-cli <command> --help' for command-specific options.")
}
