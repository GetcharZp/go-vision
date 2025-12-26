package vision

import (
	"fmt"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
	"image"
	"image/color"
	"image/draw"
	"os"
)

// TextDrawer 文本绘制工具
type TextDrawer struct {
	font     *opentype.Font
	face     font.Face
	fontSize float64
}

// NewTextDrawer 创建文本绘制工具
//
// # Params:
//
//	fontPath: 字体路径
func NewTextDrawer(fontPath string) (*TextDrawer, error) {
	fontBytes, err := os.ReadFile(fontPath)
	if err != nil {
		return nil, fmt.Errorf("打开字体文件失败：%w", err)
	}

	ttFont, err := opentype.Parse(fontBytes)
	if err != nil {
		return nil, fmt.Errorf("解析字体文件失败：%w", err)
	}

	d := &TextDrawer{font: ttFont}
	if err := d.SetSize(12); err != nil {
		return nil, err
	}
	return d, nil
}

// SetSize 动态调整字体大小
//
// # Params:
//
//	fontSize: 字体大小
func (d *TextDrawer) SetSize(fontSize float64) error {
	if d.face != nil && d.fontSize == fontSize {
		return nil
	}

	// 释放旧 Face 内存
	if d.face != nil {
		d.face.Close()
	}

	nf, err := opentype.NewFace(d.font, &opentype.FaceOptions{
		Size:    fontSize,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		return err
	}

	d.face = nf
	d.fontSize = fontSize
	return nil
}

// DrawText 绘制文本
//
// # Params:
//
//	img: 被绘制的图像
//	text: 绘制的文本
//	x, y: 绘制的坐标
//	c: 绘制的颜色
func (d *TextDrawer) DrawText(img draw.Image, text string, x, y int, c color.Color) {
	point := fixed.Point26_6{
		X: fixed.I(x),
		Y: fixed.I(y),
	}

	d1 := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c), // 文字颜色源
		Face: d.face,
		Dot:  point, // 开始绘制的点
	}
	d1.DrawString(text)
}

// Close 释放资源
func (d *TextDrawer) Close() {
	if d.face != nil {
		d.face.Close()
	}
}
