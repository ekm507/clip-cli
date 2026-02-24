package clip

import (
	ort "github.com/yalue/onnxruntime_go"
)

func InitializeEnvironment(sharedLibraryPath string) (func(), error) {
	ort.SetSharedLibraryPath(sharedLibraryPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, err
	}

	return func() {
		ort.DestroyEnvironment()
	}, nil
}
