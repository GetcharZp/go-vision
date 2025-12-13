package examples

import (
	"fmt"
	"github.com/getcharzp/go-vision/yolov11"
	"github.com/up-zero/gotool/imageutil"
	"testing"
)

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
