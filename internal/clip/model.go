package clip

import (
	"clip_cli/internal/config"

	ort "github.com/yalue/onnxruntime_go"
)

type Runner struct {
	cfg config.Config
}

func NewRunner(cfg config.Config) Runner {
	return Runner{cfg: cfg}
}

func (r Runner) RunVisionModel(inputData []float32) ([]float32, error) {
	inShape := ort.NewShape(1, 3, 224, 224)
	inputTensor, err := ort.NewTensor(inShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outShape := ort.NewShape(1, r.cfg.EmbeddingDim)
	outputData := make([]float32, r.cfg.EmbeddingDim)
	outputTensor, err := ort.NewTensor(outShape, outputData)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(r.cfg.VisionModelPath,
		[]string{r.cfg.VisionInputName},
		[]string{r.cfg.VisionOutputName},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, err
	}

	return outputData, nil
}

func (r Runner) RunTextModel(idsData []int64, masksData []int64) ([]float32, error) {
	inShape := ort.NewShape(1, r.cfg.MaxTokens)

	idsTensor, err := ort.NewTensor(inShape, idsData)
	if err != nil {
		return nil, err
	}
	defer idsTensor.Destroy()

	masksTensor, err := ort.NewTensor(inShape, masksData)
	if err != nil {
		return nil, err
	}
	defer masksTensor.Destroy()

	outShape := ort.NewShape(1, r.cfg.EmbeddingDim)
	outputData := make([]float32, r.cfg.EmbeddingDim)
	outputTensor, err := ort.NewTensor(outShape, outputData)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(r.cfg.TextModelPath,
		[]string{r.cfg.TextInputIDs, r.cfg.TextAttention},
		[]string{r.cfg.TextOutputName},
		[]ort.ArbitraryTensor{idsTensor, masksTensor},
		[]ort.ArbitraryTensor{outputTensor},
		nil,
	)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, err
	}

	return outputData, nil
}
