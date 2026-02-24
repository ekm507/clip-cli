package storage

import (
	"bytes"
	"database/sql"
	"encoding/binary"

	_ "github.com/mattn/go-sqlite3"
)

type Store struct {
	db *sql.DB
}

type ImageRecord struct {
	Path      string
	Embedding []float32
}

func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, err
	}

	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, embedding BLOB)`); err != nil {
		db.Close()
		return nil, err
	}

	return &Store{db: db}, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) UpsertImage(path string, embedding []float32) error {
	blob := float32ArrayToBytes(embedding)
	_, err := s.db.Exec(`INSERT OR REPLACE INTO images (path, embedding) VALUES (?, ?)`, path, blob)
	return err
}

func (s *Store) ListImages() ([]ImageRecord, error) {
	rows, err := s.db.Query(`SELECT path, embedding FROM images`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results := make([]ImageRecord, 0)
	for rows.Next() {
		var path string
		var blob []byte
		if err := rows.Scan(&path, &blob); err != nil {
			continue
		}

		results = append(results, ImageRecord{
			Path:      path,
			Embedding: bytesToFloat32Array(blob),
		})
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}

func float32ArrayToBytes(values []float32) []byte {
	buf := new(bytes.Buffer)
	_ = binary.Write(buf, binary.LittleEndian, values)
	return buf.Bytes()
}

func bytesToFloat32Array(data []byte) []float32 {
	values := make([]float32, len(data)/4)
	buf := bytes.NewReader(data)
	_ = binary.Read(buf, binary.LittleEndian, &values)
	return values
}
