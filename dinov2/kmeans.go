package dinov2

import (
	"math"
	"math/rand"
)

// kmeans 在 [n, dim] 的特征矩阵上执行 k-means 聚类 (k-means++ 初始化, Lloyd 迭代)
//
// # Params:
//
//	features: 按行存储的特征矩阵, 长度 n*dim
//	n: 样本数量
//	dim: 特征维度
//	k: 聚类数量
//	maxIter: 最大迭代次数
//
// # Returns:
//
//	每个样本所属的聚类标签 (0 ~ k-1)
func kmeans(features []float32, n, dim, k, maxIter int) []int {
	if k <= 0 {
		k = 1
	}
	if k > n {
		k = n
	}

	labels := make([]int, n)
	centroids := make([]float32, k*dim)

	// k-means++ 初始化质心
	rng := rand.New(rand.NewSource(42)) // 固定种子保证可复现

	// 随机选取第一个质心
	first := rng.Intn(n)
	copy(centroids[0:dim], features[first*dim:first*dim+dim])

	for c := 1; c < k; c++ {
		minDists := make([]float32, n)
		var sum float32
		for i := 0; i < n; i++ {
			best := float32(math.MaxFloat32)
			for j := 0; j < c; j++ {
				dist := sqDist(features[i*dim:i*dim+dim], centroids[j*dim:j*dim+dim], dim)
				if dist < best {
					best = dist
				}
			}
			minDists[i] = best
			sum += best
		}

		// 按距离加权随机选择下一个质心
		r := rng.Float32() * sum
		var acc float32
		chosen := n - 1
		for i := 0; i < n; i++ {
			acc += minDists[i]
			if acc >= r {
				chosen = i
				break
			}
		}
		copy(centroids[c*dim:c*dim+dim], features[chosen*dim:chosen*dim+dim])
	}

	// Lloyd 迭代
	for iter := 0; iter < maxIter; iter++ {
		changed := false

		// E 步: 分配样本到最近的质心
		for i := 0; i < n; i++ {
			best := 0
			bestDist := float32(math.MaxFloat32)
			for j := 0; j < k; j++ {
				dist := sqDist(features[i*dim:i*dim+dim], centroids[j*dim:j*dim+dim], dim)
				if dist < bestDist {
					bestDist = dist
					best = j
				}
			}
			if labels[i] != best {
				labels[i] = best
				changed = true
			}
		}

		// 标签不再变化, 提前收敛
		if !changed {
			break
		}

		// M 步: 重新计算质心
		counts := make([]int, k)
		newCentroids := make([]float32, k*dim)
		for i := 0; i < n; i++ {
			c := labels[i]
			counts[c]++
			base := c * dim
			fi := i * dim
			for j := 0; j < dim; j++ {
				newCentroids[base+j] += features[fi+j]
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				base := c * dim
				for j := 0; j < dim; j++ {
					newCentroids[base+j] /= float32(counts[c])
				}
			}
		}
		centroids = newCentroids
	}

	return labels
}

// sqDist 计算两个 dim 维向量的欧氏距离平方
func sqDist(a, b []float32, dim int) float32 {
	var sum float32
	for j := 0; j < dim; j++ {
		d := a[j] - b[j]
		sum += d * d
	}
	return sum
}
