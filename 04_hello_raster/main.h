// MIT License
// 
// Copyright (C) 2018-2024, Tellusim Technologies Inc. https://tellusim.com/
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef __MAIN_H__
#define __MAIN_H__

/*
 */
struct Vertex {
	Vector4f position;
	Vector4f normal;
};

/*
 */
struct GeometryParameters {
	Vector4f bound_min;				// bound box minimum
	Vector4f bound_max;				// bound box maxumum
	float32_t error;				// visibility error
	uint32_t parent_0;				// first parent index
	uint32_t parent_1;				// second parent index
	uint32_t num_children;			// number of children geometries
	uint32_t base_child;			// base child
	uint32_t num_vertices;			// number of vertices
	uint32_t base_vertex;			// base vertex
	uint32_t num_primitives;		// number of primitives
	uint32_t base_primitive;		// base primitive
	uint32_t padding[3];
};

#endif /* __MAIN_H__ */
