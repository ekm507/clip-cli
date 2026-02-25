package storage

import (
	"C"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
)

func init() {
	sqlite_vec.Auto()
}
