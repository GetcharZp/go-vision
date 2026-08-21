package dinov2

import (
	"image"
	"image/color"
	"image/draw"
)

// palette 用于可视化不同分割区域的颜色
var palette = []color.RGBA{
	{R: 255, G: 0, B: 0, A: 255},     // 红
	{G: 255, A: 255},                 // 绿
	{B: 255, A: 255},                 // 蓝
	{R: 255, G: 255, A: 255},         // 黄
	{R: 255, G: 0, B: 255, A: 255},   // 品红
	{R: 0, G: 255, B: 255, A: 255},   // 青
	{R: 255, G: 165, B: 0, A: 255},   // 橙
	{R: 128, G: 0, B: 128, A: 255},   // 紫
	{R: 0, G: 128, B: 128, A: 255},   // 蓝绿
	{R: 255, G: 105, B: 180, A: 255}, // 粉
}

// DrawResult 将分割结果以不同颜色半透明叠加到原图上
//
// # Params:
//
//	img: 原图
//	result: 分割结果
func DrawResult(img image.Image, result *Result) image.Image {
	bounds := img.Bounds()
	dst := image.NewRGBA(bounds)
	draw.Draw(dst, bounds, img, bounds.Min, draw.Src)

	const alpha = 0.6 // 叠加透明度

	for y := 0; y < result.Height; y++ {
		for x := 0; x < result.Width; x++ {
			label := int(result.Labels.GrayAt(x, y).Y)
			if label < 0 || label >= len(palette) {
				continue
			}
			c := palette[label]

			// 对应到原图坐标
			px := bounds.Min.X + x
			py := bounds.Min.Y + y
			base := dst.RGBAAt(px, py)

			dst.SetRGBA(px, py, color.RGBA{
				R: uint8(float32(base.R)*(1-alpha) + float32(c.R)*alpha),
				G: uint8(float32(base.G)*(1-alpha) + float32(c.G)*alpha),
				B: uint8(float32(base.B)*(1-alpha) + float32(c.B)*alpha),
				A: 255,
			})
		}
	}
	return dst
}
