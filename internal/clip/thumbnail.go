package clip

import (
	"bytes"
	"fmt"
	"image"
	"os"

	_ "image/jpeg"
	_ "image/png"

	"github.com/disintegration/imaging"
)

// GenerateThumbnailJPEG creates a square JPEG thumbnail with the given side size.
func GenerateThumbnailJPEG(imagePath string, size int) ([]byte, error) {
	if size <= 0 {
		return nil, fmt.Errorf("thumbnail size must be > 0")
	}

	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	thumb := imaging.Fill(img, size, size, imaging.Center, imaging.CatmullRom)

	buf := new(bytes.Buffer)
	if err := imaging.Encode(buf, thumb, imaging.JPEG, imaging.JPEGQuality(85)); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}
