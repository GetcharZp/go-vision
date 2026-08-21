package examples

import (
	"fmt"
	"testing"

	"github.com/getcharzp/go-vision/dinov2"
	"github.com/up-zero/gotool/imageutil"
)

// TestDINOv2Seg DINOv2 无监督分割
//
// DINOv2 输出的是 Patch 特征 (last_hidden_state), 通过 k-means 聚类将
// 特征相似的区域划分为同一个分割区域, 属于无监督分割。
func TestDINOv2Seg(t *testing.T) {
	cfg := dinov2.DefaultConfig()
	cfg.ModelPath = "../dinov2_weights/model.onnx"
	cfg.OnnxRuntimeLibPath = "../lib/onnxruntime.dll"
	cfg.NumClusters = 3

	engine, err := dinov2.NewEngine(cfg)
	if err != nil {
		t.Fatalf("初始化引擎失败: %v", err)
	}
	defer engine.Destroy()

	img, err := imageutil.Open("./test.png")
	if err != nil {
		t.Fatalf("打开图片失败: %v", err)
	}

	result, err := engine.Predict(img)
	if err != nil {
		t.Fatalf("预测失败: %v", err)
	}

	fmt.Printf("原图尺寸: %d x %d, 分割区域: %d 个\n", result.Width, result.Height, len(result.Masks))
	for idx, mask := range result.Masks {
		// 统计该区域的像素占比
		count := 0
		for _, v := range mask.Pix {
			if v == 255 {
				count++
			}
		}
		fmt.Printf("区域 %d 像素数: %d (%.1f%%)\n", idx, count, float64(count)*100/float64(result.Width*result.Height))
		imageutil.Save(fmt.Sprintf("dinov2_seg_mask_%d.png", idx), mask, 100)
	}

	dst := dinov2.DrawResult(img, result)
	imageutil.Save("dinov2_seg.jpg", dst, 90)
}
