package clip

import (
	"image"
	"os"

	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"
	"github.com/sugarme/tokenizer/pretrained"
)

func ProcessImageExact(imagePath string) ([]float32, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	resized := imaging.Fill(img, 224, 224, imaging.Center, imaging.CatmullRom)
	mean := []float32{0.48145466, 0.4578275, 0.40821073}
	std := []float32{0.26862954, 0.26130258, 0.27577711}
	output := make([]float32, 3*224*224)

	for y := 0; y < 224; y++ {
		for x := 0; x < 224; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()

			fr := float32(r) / 65535.0
			fg := float32(g) / 65535.0
			fb := float32(b) / 65535.0

			fr = (fr - mean[0]) / std[0]
			fg = (fg - mean[1]) / std[1]
			fb = (fb - mean[2]) / std[2]

			output[0*224*224+y*224+x] = fr
			output[1*224*224+y*224+x] = fg
			output[2*224*224+y*224+x] = fb
		}
	}

	return output, nil
}

func ProcessText(text, tokenizerPath string, maxTokens int64) ([]int64, []int64, error) {
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, nil, err
	}

	encoded, err := tk.EncodeSingle(text)
	if err != nil {
		return nil, nil, err
	}

	ids := make([]int64, maxTokens)
	masks := make([]int64, maxTokens)

	for i := 0; i < len(encoded.Ids) && i < int(maxTokens); i++ {
		ids[i] = int64(encoded.Ids[i])
		masks[i] = int64(encoded.AttentionMask[i])
	}

	return ids, masks, nil
}
