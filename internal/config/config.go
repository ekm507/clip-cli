package config

// Config groups the runtime configuration for the CLI.
type Config struct {
	VisionModelPath  string
	TextModelPath    string
	TokenizerPath    string
	DatabasePath     string
	ORTSharedLibPath string
	EmbeddingDim     int64
	MaxTokens        int64
	VisionInputName  string
	VisionOutputName string
	TextInputIDs     string
	TextAttention    string
	TextOutputName   string
}

// Default returns default values used by the local project layout.
func Default() Config {
	return Config{
		VisionModelPath:  "models/vision.onnx",
		TextModelPath:    "models/text.onnx",
		TokenizerPath:    "models/tokenizer.json",
		DatabasePath:     "smart_gallery.db",
		ORTSharedLibPath: "/usr/lib/libonnxruntime.so",
		EmbeddingDim:     512,
		MaxTokens:        77,
		VisionInputName:  "pixel_values",
		VisionOutputName: "image_embeds",
		TextInputIDs:     "input_ids",
		TextAttention:    "attention_mask",
		TextOutputName:   "sentence_embeddings",
	}
}
