package examples

import (
	"fmt"
	"github.com/getcharzp/go-vision/yolov11"
	"github.com/up-zero/gotool/imageutil"
	"image"
	"image/color"
	"image/draw"
	"testing"
)

func TestYOLOv11Det(t *testing.T) {
	cfg := yolov11.DefaultDetConfig()
	cfg.ModelPath = "../yolov11_weights/yolo11m.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolov11.NewDetEngine(cfg)
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
	imageutil.Save("yolov11_det.jpg", targetImg, 50)
}

func TestYOLOv11Seg(t *testing.T) {
	cfg := yolov11.DefaultSegConfig()
	cfg.ModelPath = "../yolov11_weights/yolo11m-seg.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolov11.NewSegEngine(cfg)
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
		imageutil.Save(fmt.Sprintf("yolov11_seg_mask_%d.png", idx), res.Mask, 100)
	}
}

func TestYOLOv11Cls(t *testing.T) {
	cfg := yolov11.DefaultClsConfig()
	cfg.ModelPath = "../yolov11_weights/yolo11m-cls.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolov11.NewClsEngine(cfg)
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

func TestYOLOv11Pose(t *testing.T) {
	cfg := yolov11.DefaultPoseConfig()
	cfg.ModelPath = "../yolov11_weights/yolo11m-pose.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"

	engine, err := yolov11.NewPoseEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./person.jpg")
	results, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	dst := yolov11.DrawPoseResult(img, results)
	imageutil.Save("yolov11_pose.jpg", dst, 50)
}
