// MIT License
// 
// Copyright (C) 2018-2023, Tellusim Technologies Inc. https://tellusim.com/
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
struct ComputeParameters {
	Vector4f camera;
	Vector4f global_gravity;
	Vector4f wind_velocity;
	uint32_t num_emitters;
	uint32_t max_emitters;
	uint32_t max_particles;
	float32_t wind_force;
	float32_t velocity_damping;
	float32_t growth_damping;
	float32_t twist_damping;
	float32_t ifps;
};

/*
 */
struct ComputeState {
	uint32_t num_particles;
	uint32_t new_particles;
	uint32_t padding[2];
};

/*
 */
struct EmitterParameters {
	Vector4f position;
	Vector4f direction;
	Color color_mean;
	Color color_spread;
	float32_t position_mean;
	float32_t position_spread;
	float32_t velocity_mean;
	float32_t velocity_spread;
	float32_t velocity_damping;
	float32_t radius_mean;
	float32_t radius_spread;
	float32_t growth_mean;
	float32_t growth_spread;
	float32_t growth_damping;
	float32_t angle_mean;
	float32_t angle_spread;
	float32_t twist_mean;
	float32_t twist_spread;
	float32_t twist_damping;
	float32_t life_mean;
	float32_t life_spread;
	float32_t spawn_mean;
	float32_t spawn_spread;
	uint32_t padding;
};

/*
 */
struct EmitterState {
	Vector4f position;
	Vector2i seed;
	float32_t spawn;
	uint32_t padding;
};

/*
 */
struct ParticleState {
	Vector4f position;
	Vector4f color;
	Vector4f velocity;
	float32_t velocity_damping;
	float32_t radius;
	float32_t growth;
	float32_t growth_damping;
	float32_t angle;
	float32_t twist;
	float32_t twist_damping;
	float32_t life;
	float32_t time;
	uint32_t padding[3];
};

/*
 */
struct Vertex {
	Vector4f position;
	Vector4f velocity;
	Vector4f color;
};

#endif /* __MAIN_H__ */
