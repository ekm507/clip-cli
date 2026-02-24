package vector

import "math"

func L2Normalize(values []float32) {
	var sum float32
	for _, value := range values {
		sum += value * value
	}

	norm := float32(math.Sqrt(float64(sum)))
	if norm == 0 {
		return
	}

	for i := range values {
		values[i] /= norm
	}
}

func DotProduct(a, b []float32) float32 {
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}
