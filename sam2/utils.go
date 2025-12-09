package sam2

import (
	"image"
)

// normalizeAndPad 归一化和填充
func normalizeAndPad(src image.Image, targetW, targetH int) []float32 {
	bounds := src.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	data := make([]float32, 3*targetW*targetH)

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := src.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			// RGBA returns 0-65535
			rf := float32(r) / 65535.0
			gf := float32(g) / 65535.0
			bf := float32(b) / 65535.0

			rf = (rf - MeanR) / StdR
			gf = (gf - MeanG) / StdG
			bf = (bf - MeanB) / StdB

			// 目标索引 (CHW)
			idx := y*targetW + x
			data[idx] = rf
			data[targetW*targetH+idx] = gf
			data[2*targetW*targetH+idx] = bf
		}
	}
	return data
}

// upscaleMaskLogits 原图尺寸的预测结果
func upscaleMaskLogits(logits []float32, logitsDim, validW, validH, dstW, dstH int) []uint8 {
	output := make([]uint8, dstW*dstH)
	xRatio := float32(validW) / float32(dstW)
	yRatio := float32(validH) / float32(dstH)

	for y := 0; y < dstH; y++ {
		srcY := int(float32(y) * yRatio)
		if srcY >= validH {
			srcY = validH - 1
		}
		for x := 0; x < dstW; x++ {
			srcX := int(float32(x) * xRatio)
			if srcX >= validW {
				srcX = validW - 1
			}

			val := logits[srcY*logitsDim+srcX]
			if val > maskThreshold {
				output[y*dstW+x] = 255
			} else {
				output[y*dstW+x] = 0
			}
		}
	}
	return output
}
