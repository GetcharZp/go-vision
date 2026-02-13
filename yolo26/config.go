package yolo26

import (
	"github.com/getcharzp/go-vision"
	"image"
)

// Config 引擎的初始化参数
type Config struct {
	ModelPath          string // ONNX 模型路径
	OnnxRuntimeLibPath string // ONNX Runtime 动态库路径

	// 推理参数
	ConfThreshold float32 // 置信度阈值 (默认 0.45)
	MaskThreshold float32 // Mask 二值化阈值 (默认 0.5)

	// 模型参数
	InputSize     int // 默认 640
	NumClasses    int // 默认 80
	NumMaskCoeffs int // 默认 32
	NumKeyPoints  int // 默认 17

	// 可选参数
	UseCuda           bool // (可选) 是否启用 CUDA
	NumThreads        int  // (可选) ONNX 线程数, 默认由CPU核心数决定
	EnableCpuMemArena bool // (可选) 是否开启 ONNX 内存池
}

// DetResult 目标检测结果
type DetResult struct {
	// 分类ID，例如：
	//	0: person
	//  1: bicycle
	//  2: car
	// - 详细映射参考：
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
	ClassID int
	Score   float32
	Box     image.Rectangle // 检测框
}

// DefaultConfig 默认配置
func DefaultConfig() Config {
	return Config{
		OnnxRuntimeLibPath: vision.DefaultLibraryPath(),
		ConfThreshold:      0.45,
		MaskThreshold:      0.50,
		InputSize:          640,
		NumClasses:         80,
		NumMaskCoeffs:      32,
		NumKeyPoints:       17,
	}
}

// DefaultDetConfig 检测的默认配置
func DefaultDetConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = "./yolo26_weights/yolo26m.onnx"
	return cfg
}

// imageParams 图片尺寸信息
type imageParams struct {
	origW, origH int
	scale        float32
}

// DefaultSegConfig 分割的默认配置
func DefaultSegConfig() Config {
	cfg := DefaultConfig()
	cfg.ModelPath = "./yolo26_weights/yolo26m-seg.onnx"
	return cfg
}

// SegResult 分割结果
type SegResult struct {
	// 分类ID，例如：
	//	0: person
	//  1: bicycle
	//  2: car
	// 详细映射参考：
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
	ClassID int
	Score   float32
	Box     image.Rectangle // 分割出的矩形区域
	Mask    *image.Gray     // 解码后的 Mask
}

// ClassResult 分类结果
type ClassResult struct {
	// 分类ID，例如：
	//	436: station wagon
	//	656: minivan
	// 详细映射参考：
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/ImageNet.yaml
	ClassID int
	Score   float32
}

// DefaultClsConfig 分类的默认配置
func DefaultClsConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 224
	cfg.ModelPath = "./yolo26_weights/yolo26-cls.onnx"
	return cfg
}

// KeyPoint 单个关键点
type KeyPoint struct {
	X, Y  int     // 原图坐标
	Score float32 // 可见性/置信度
}

// PoseResult 姿态估计结果
type PoseResult struct {
	//	https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco-pose.yaml
	ClassID   int
	Score     float32
	Box       image.Rectangle
	KeyPoints []KeyPoint // 关键点列表
}

// DefaultPoseConfig 姿势的默认配置
func DefaultPoseConfig() Config {
	cfg := DefaultConfig()
	cfg.NumClasses = 1
	cfg.ModelPath = "./yolo26_weights/yolo26m-pose.onnx"
	return cfg
}

// OBBResult 旋转目标检测结果
type OBBResult struct {
	ClassID int
	Score   float32
	// 旋转框的顶点坐标：TopLeft, TopRight, BottomRight, BottomLeft
	Corners [4]image.Point

	Center image.Point
	Angle  float32 // 弧度
}

// DefaultOBBConfig OBB的默认配置
func DefaultOBBConfig() Config {
	cfg := DefaultConfig()
	cfg.InputSize = 1024
	cfg.NumClasses = 15
	cfg.ModelPath = "./yolo26_weights/yolo26m-obb.onnx"
	return cfg
}
