package examples

import (
	"fmt"
	"github.com/getcharzp/go-vision/yolo26"
	"github.com/up-zero/gotool/imageutil"
	"image"
	"image/color"
	"image/draw"
	"testing"
)

func TestYOLO26Det(t *testing.T) {
	cfg := yolo26.DefaultDetConfig()
	cfg.ModelPath = "../yolo26_weights/yolo26m.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolo26.NewDetEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	targetImg := image.NewRGBA(img.Bounds())
	draw.Draw(targetImg, img.Bounds(), img, img.Bounds().Min, draw.Src)
	fmt.Printf("检测到目标: %d 个\n", len(results))
	for _, res := range results {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		imageutil.DrawThickRectOutline(targetImg, res.Box, color.RGBA{R: 255, G: 0, B: 0, A: 255}, 3)
	}
	imageutil.Save("yolo26_det.jpg", targetImg, 50)
}

func TestYOLO26Seg(t *testing.T) {
	cfg := yolo26.DefaultSegConfig()
	cfg.ModelPath = "../yolo26_weights/yolo26s-seg.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolo26.NewSegEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./test.png")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	fmt.Printf("检测到目标: %d 个\n", len(results))
	for idx, res := range results {
		fmt.Printf("Class: %d, Score: %.2f, Box: %v\n", res.ClassID, res.Score, res.Box)
		imageutil.Save(fmt.Sprintf("yolo26_seg_mask_%d.png", idx), res.Mask, 100)
	}
}

func TestYOLO26Cls(t *testing.T) {
	cfg := yolo26.DefaultClsConfig()
	cfg.ModelPath = "../yolo26_weights/yolo26m-cls.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolo26.NewClsEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./test.png")
	results, err := engine.Predict(img, 5)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	for _, res := range results {
		fmt.Printf("Class: %d, Score: %.5f\n", res.ClassID, res.Score)
	}
}

func TestYOLO26Pose(t *testing.T) {
	cfg := yolo26.DefaultPoseConfig()
	cfg.ModelPath = "../yolo26_weights/yolo26m-pose.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolo26.NewPoseEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./person.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	dst := yolo26.DrawPoseResult(img, results)
	imageutil.Save("yolo26_pose.jpg", dst, 50)
}

func TestYOLO26OBB(t *testing.T) {
	cfg := yolo26.DefaultOBBConfig()
	cfg.ModelPath = "../yolo26_weights/yolo26m-obb.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolo26.NewOBBEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./ship.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	dst := image.NewRGBA(img.Bounds())
	draw.Draw(dst, img.Bounds(), img, img.Bounds().Min, draw.Src)
	for _, result := range results {
		imageutil.DrawThickPolygonOutline(dst, result.Corners[:], 3, color.RGBA{R: 255, G: 0, B: 0, A: 255})
	}
	imageutil.Save("yolo26_obb.jpg", dst, 50)
}
