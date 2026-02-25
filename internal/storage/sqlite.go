package storage

import (
	"database/sql"
	"fmt"
	"strconv"
	"strings"

	_ "github.com/mattn/go-sqlite3"
)

type Store struct {
	db *sql.DB
}

type SearchResult struct {
	Path     string
	Distance float32
}

func Open(path string, embeddingDim int64) (*Store, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, err
	}

	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE)`); err != nil {
		db.Close()
		return nil, err
	}

	createVecTableQuery := fmt.Sprintf(
		`CREATE VIRTUAL TABLE IF NOT EXISTS image_embeddings USING vec0(embedding float[%d])`,
		embeddingDim,
	)
	if _, err := db.Exec(createVecTableQuery); err != nil {
		db.Close()
		return nil, err
	}

	return &Store{db: db}, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}

func (s *Store) UpsertImage(path string, embedding []float32) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}

	if _, err := tx.Exec(`INSERT OR IGNORE INTO images (path) VALUES (?)`, path); err != nil {
		tx.Rollback()
		return err
	}

	var imageID int64
	if err := tx.QueryRow(`SELECT id FROM images WHERE path = ?`, path).Scan(&imageID); err != nil {
		tx.Rollback()
		return err
	}

	if _, err := tx.Exec(`DELETE FROM image_embeddings WHERE rowid = ?`, imageID); err != nil {
		tx.Rollback()
		return err
	}

	if _, err := tx.Exec(
		`INSERT INTO image_embeddings(rowid, embedding) VALUES(?, ?)`,
		imageID,
		toVecJSON(embedding),
	); err != nil {
		tx.Rollback()
		return err
	}

	return tx.Commit()
}

func (s *Store) SearchByEmbedding(embedding []float32, limit int) ([]SearchResult, error) {
	if limit < 1 {
		limit = 1
	}

	rows, err := s.db.Query(
		`SELECT images.path, image_embeddings.distance
		 FROM image_embeddings
		 JOIN images ON images.id = image_embeddings.rowid
		 WHERE embedding MATCH ? AND k = ?
		 ORDER BY distance`,
		toVecJSON(embedding),
		limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	results := make([]SearchResult, 0, limit)
	for rows.Next() {
		var result SearchResult
		if err := rows.Scan(&result.Path, &result.Distance); err != nil {
			return nil, err
		}
		results = append(results, result)
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}

func toVecJSON(values []float32) string {
	var builder strings.Builder
	builder.WriteByte('[')
	for i, value := range values {
		if i > 0 {
			builder.WriteByte(',')
		}
		builder.WriteString(strconv.FormatFloat(float64(value), 'f', -1, 32))
	}
	builder.WriteByte(']')
	return builder.String()
}
