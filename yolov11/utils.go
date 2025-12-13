package yolov11

import (
	"image"
	"math"
	"sort"
)

func sigmoid(x float32) float32 {
	return 1.0 / (1.0 + float32(math.Exp(float64(-x))))
}

// nms 非极大值抑制，过滤掉重叠度过高的检测框
func nms(dets []segInternal, iouThresh float32) []int {
	sort.Slice(dets, func(i, j int) bool {
		return dets[i].score > dets[j].score
	})

	keep := make([]int, 0)
	suppressed := make([]bool, len(dets))

	for i := 0; i < len(dets); i++ {
		if suppressed[i] {
			continue
		}
		keep = append(keep, i)

		for j := i + 1; j < len(dets); j++ {
			if suppressed[j] {
				continue
			}
			if computeIOU(dets[i].origBox, dets[j].origBox) > iouThresh {
				suppressed[j] = true
			}
		}
	}
	return keep
}

func computeIOU(r1, r2 image.Rectangle) float32 {
	intersect := r1.Intersect(r2)
	if intersect.Empty() {
		return 0.0
	}

	interArea := intersect.Dx() * intersect.Dy()
	area1 := r1.Dx() * r1.Dy()
	area2 := r2.Dx() * r2.Dy()

	return float32(interArea) / float32(area1+area2-interArea)
}
