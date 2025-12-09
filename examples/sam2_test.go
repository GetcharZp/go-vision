package examples

import (
	"fmt"
	"github.com/getcharzp/go-vision/sam2"
	"github.com/up-zero/gotool/imageutil"
	_ "image/jpeg"
	"testing"
)

func TestSAM2Refactored(t *testing.T) {
	config := sam2.Config{
		OnnxRuntimeLibPath: "../lib/onnxruntime.dll",
		EncodeModelPath:    "../sam2_weights/vision_encoder.onnx",
		DecodeModelPath:    "../sam2_weights/prompt_encoder_mask_decoder.onnx",
	}

	engine, err := sam2.NewEngine(config)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, _ := imageutil.Open("./test.png")
	imgCtx, err := engine.EncodeImage(img)
	if err != nil {
		t.Fatalf("图片 Encode 失败: %v", err)
	}
	defer imgCtx.Destroy()

	points := []sam2.Point{
		{X: 367, Y: 168, Label: sam2.LabelBoxTopLeft},  // 左上
		{X: 441, Y: 349, Label: sam2.LabelBoxBotRight}, // 右下
	}
	imgResult, score, err := imgCtx.Decode(points)
	if err != nil {
		t.Fatalf("Mask Decode 失败: %v", err)
	}

	fmt.Printf("Mask generated, score: %.4f\n", score)
	imageutil.Save("output_mask.png", imgResult, 100)
}
