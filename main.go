package main

import (
	"bytes"
	"database/sql"
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"sort"

	"github.com/disintegration/imaging"
	_ "github.com/mattn/go-sqlite3"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

// ==========================================
// تنظیمات پایه (شما می‌توانید این مقادیر را تغییر دهید)
// ==========================================
const (
	visionModelPath  = "models/vision.onnx"
	textModelPath    = "models/text.onnx"
	tokenizerPath    = "models/tokenizer.json"
	dbPath           = "smart_gallery.db"
	ortSharedLibPath = "./libonnxruntime.so" // در ویندوز: onnxruntime.dll
	embedDim         = 512                   // طول بردار خروجی CLIP (معمولا 512 یا 768)
	maxTokens        = 77                    // طول استاندارد توکن‌های CLIP
)

// نام ورودی و خروجی‌های مدل در ONNX (اگر مدل شما نام متفاوتی دارد اینجا تغییر دهید)
const (
	visionInputName  = "pixel_values"
	visionOutputName = "output" // در بعضی مدل‌ها: image_embeds
	textInputIds     = "input_ids"
	textAttention    = "attention_mask"
	textOutputName   = "output" // در بعضی مدل‌ها: text_embeds
)

func main() {
	// راه‌اندازی ONNX Runtime
	ort.SetSharedLibraryPath(ortSharedLibPath)
	err := ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("خطا در راه‌اندازی ONNX Runtime: %v\n(آیا فایل کتابخانه مشترک را در مسیر درست قرار داده‌اید؟)", err)
	}
	defer ort.DestroyEnvironment()

	// راه‌اندازی دیتابیس SQLite
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatal("خطا در باز کردن دیتابیس:", err)
	}
	defer db.Close()

	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT UNIQUE, embedding BLOB)`)
	if err != nil {
		log.Fatal("خطا در ساخت جدول:", err)
	}

	// تعریف دستورات CLI
	addCmd := flag.NewFlagSet("add", flag.ExitOnError)
	addImgPath := addCmd.String("image", "", "Path to the image file")

	searchCmd := flag.NewFlagSet("search", flag.ExitOnError)
	searchText := searchCmd.String("text", "", "Text query to search for")
	searchLimit := searchCmd.Int("limit", 5, "Number of results to show")

	if len(os.Args) < 2 {
		fmt.Println("راهنما: ./smart-gallery [add|search] [گزینه‌ها]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "add":
		addCmd.Parse(os.Args[2:])
		if *addImgPath == "" {
			fmt.Println("لطفا مسیر عکس را وارد کنید: --image path/to/img.jpg")
			return
		}
		handleAdd(db, *addImgPath)

	case "search":
		searchCmd.Parse(os.Args[2:])
		if *searchText == "" {
			fmt.Println("لطفا متن جستجو را وارد کنید: --text \"your query\"")
			return
		}
		handleSearch(db, *searchText, *searchLimit)

	default:
		fmt.Println("دستور نامعتبر. از add یا search استفاده کنید.")
		os.Exit(1)
	}
}

// ==========================================
// منطق دستورات CLI
// ==========================================

func handleAdd(db *sql.DB, imgPath string) {
	fmt.Println("در حال پردازش تصویر...")
	pixelValues, err := processImageExact(imgPath)
	if err != nil {
		log.Fatal("خطا در پردازش تصویر:", err)
	}

	fmt.Println("در حال استخراج بردار از مدل Vision...")
	embedding, err := runVisionModel(pixelValues)
	if err != nil {
		log.Fatal("خطا در اجرای مدل:", err)
	}

	l2Normalize(embedding)
	blob := float32ArrayToBytes(embedding)

	_, err = db.Exec(`INSERT OR REPLACE INTO images (path, embedding) VALUES (?, ?)`, imgPath, blob)
	if err != nil {
		log.Fatal("خطا در ذخیره در دیتابیس:", err)
	}
	fmt.Printf("تصویر %s با موفقیت اضافه شد.\n", imgPath)
}

func handleSearch(db *sql.DB, query string, limit int) {
	fmt.Println("در حال توکنایز کردن متن...")
	ids, masks, err := processText(query)
	if err != nil {
		log.Fatal("خطا در توکنایز متن:", err)
	}

	fmt.Println("در حال استخراج بردار از مدل Text...")
	queryEmb, err := runTextModel(ids, masks)
	if err != nil {
		log.Fatal("خطا در اجرای مدل متن:", err)
	}
	l2Normalize(queryEmb)

	rows, err := db.Query(`SELECT path, embedding FROM images`)
	if err != nil {
		log.Fatal("خطا در خواندن دیتابیس:", err)
	}
	defer rows.Close()

	type SearchResult struct {
		Path  string
		Score float32
	}
	var results []SearchResult

	for rows.Next() {
		var path string
		var blob []byte
		if err := rows.Scan(&path, &blob); err != nil {
			continue
		}
		imgEmb := bytesToFloat32Array(blob)
		score := dotProduct(queryEmb, imgEmb)
		results = append(results, SearchResult{Path: path, Score: score})
	}

	// مرتب‌سازی بر اساس شباهت (از بیشترین به کمترین)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	fmt.Printf("\n--- نتایج جستجو برای: \"%s\" ---\n", query)
	for i := 0; i < limit && i < len(results); i++ {
		fmt.Printf("%d. شباهت: %.4f | مسیر: %s\n", i+1, results[i].Score, results[i].Path)
	}
}

// ==========================================
// اجرای مدل‌های ONNX
// ==========================================

func runVisionModel(inputData []float32) ([]float32, error) {
	inShape := ort.NewShape(1, 3, 224, 224)
	inputTensor, err := ort.NewTensor(inShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outShape := ort.NewShape(1, embedDim)
	outputData := make([]float32, embedDim)
	outputTensor, err := ort.NewTensor(outShape, outputData)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(visionModelPath,
		[]string{visionInputName}, []string{visionOutputName},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, err
	}
	return outputData, nil
}

func runTextModel(idsData []int64, masksData []int64) ([]float32, error) {
	inShape := ort.NewShape(1, maxTokens)
	
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

	outShape := ort.NewShape(1, embedDim)
	outputData := make([]float32, embedDim)
	outputTensor, err := ort.NewTensor(outShape, outputData)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	session, err := ort.NewAdvancedSession(textModelPath,
		[]string{textInputIds, textAttention}, []string{textOutputName},
		[]ort.ArbitraryTensor{idsTensor, masksTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		return nil, err
	}
	return outputData, nil
}

// ==========================================
// پیش‌پردازش تصویر (دقیقا مطابق پایتون)
// ==========================================

func processImageExact(imagePath string) ([]float32, error) {
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	// تغییر اندازه و برش از مرکز به سایز 224x224
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

			// تبدیل به قالب CHW
			output[0*224*224+y*224+x] = fr
			output[1*224*224+y*224+x] = fg
			output[2*224*224+y*224+x] = fb
		}
	}
	return output, nil
}

// ==========================================
// پیش‌پردازش متن (Tokenizer)
// ==========================================

func processText(text string) ([]int64, []int64, error) {
	tk, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return nil, nil, err
	}

	en, err := tk.EncodeSingle(text)
	if err != nil {
		return nil, nil, err
	}

	ids := make([]int64, maxTokens)
	masks := make([]int64, maxTokens)

	// کپی کردن توکن‌ها و پدینگ (پر کردن مابقی با صفر)
	for i := 0; i < len(en.Ids) && i < maxTokens; i++ {
		ids[i] = int64(en.Ids[i])
		masks[i] = int64(en.AttentionMask[i])
	}
	
	return ids, masks, nil
}

// ==========================================
// توابع ریاضی و دیتابیس (L2 Norm, Dot Product)
// ==========================================

func l2Normalize(vec []float32) {
	var sum float32 = 0
	for _, v := range vec {
		sum += v * v
	}
	norm := float32(math.Sqrt(float64(sum)))
	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}
}

func dotProduct(a, b []float32) float32 {
	var dot float32 = 0
	for i := range a {
		dot += a[i] * b[i]
	}
	return dot
}

func float32ArrayToBytes(arr []float32) []byte {
	buf := new(bytes.Buffer)
	// استفاده از LittleEndian مطابق استاندارد رایج
	_ = binary.Write(buf, binary.LittleEndian, arr)
	return buf.Bytes()
}

func bytesToFloat32Array(b []byte) []float32 {
	arr := make([]float32, len(b)/4)
	buf := bytes.NewReader(b)
	_ = binary.Read(buf, binary.LittleEndian, &arr)
	return arr
}

