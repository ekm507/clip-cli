package main

import (
	"log"
	"os"

	"clip_cli/internal/app"
	"clip_cli/internal/config"
)

func main() {
	application := app.New(config.Default())
	if err := application.Run(os.Args[1:]); err != nil {
		log.Fatal(err)
	}
}
