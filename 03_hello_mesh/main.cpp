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

#include <TellusimApp.h>
#include <core/TellusimLog.h>
#include <core/TellusimTime.h>
#include <core/TellusimFile.h>
#include <core/TellusimAsync.h>
#include <math/TellusimMath.h>
#include <format/TellusimMesh.h>
#include <geometry/TellusimMeshGraph.h>
#include <geometry/TellusimMeshRefine.h>
#include <graphics/TellusimMeshModel.h>
#include <platform/TellusimWindow.h>
#include <platform/TellusimDevice.h>
#include <platform/TellusimPipeline.h>
#include <platform/TellusimCommand.h>

#include "panel.h"

/*
 */
using namespace Tellusim;

/*
 */
int32_t main(int32_t argc, char **argv) {
	
	// create app
	App app(argc, argv);
	if(!app.create()) return 1;
	
	// create window
	Window window(app.getPlatform(), app.getDevice());
	if(!window || !window.setSize(app.getWidth(), app.getHeight())) return 1;
	if(!window.create("03 Hello Mesh") || !window.setHidden(false)) return 1;
	window.setKeyboardPressedCallback([&](uint32_t key, uint32_t code) {
		if(key == Window::KeyEsc) window.stop();
	});
	
	// declarations
	#include "main.h"
	
	// scene size
	constexpr uint32_t grid_size = 64;
	constexpr uint32_t grid_height = 2;
	
	// mesh parameters
	constexpr uint32_t task_group_size = 32;
	constexpr uint32_t mesh_group_size = 32;
	constexpr uint32_t max_geometries = 1024;
	constexpr uint32_t max_attributes = 256;
	constexpr uint32_t max_primitives = 256;
	
	// structures
	struct CommonParameters {
		Matrix4x4f projection;			// projection matrix
		Matrix4x4f modelview;			// modelview matrix
		Vector4f planes[4];				// clipping planes
		Vector4f signs[4];				// clipping signs
		Vector4f camera;				// camera position
		uint32_t grid_size;				// grid size
		float32_t projection_scale;		// projection scale
		float32_t target_width;			// target width
		float32_t target_height;		// target height
		float32_t time;
	};
	
	// create device
	Device device(window);
	if(!device) return 1;
	
	// check mesh shader support
	if(!device.hasShader(Shader::TypeMesh)) {
		TS_LOG(Error, "mesh shader is not supported\n");
		return 0;
	}
	
	// create pipeline
	Pipeline pipeline = device.createPipeline();
	pipeline.setUniformMask(0, Shader::MaskAll);
	pipeline.setStorageMasks(0, 5, Shader::MaskTask | Shader::MaskMesh, false);
	pipeline.setColorFormat(window.getColorFormat());
	pipeline.setDepthFormat(window.getDepthFormat());
	pipeline.setDepthFunc(Pipeline::DepthFuncLess);
	pipeline.addAttribute(Pipeline::AttributePosition, FormatRGBAf32, 0, offsetof(Vertex, position), sizeof(Vertex));
	pipeline.addAttribute(Pipeline::AttributeNormal, FormatRGBAf32, 0, offsetof(Vertex, normal), sizeof(Vertex));
	if(!pipeline.loadShaderGLSL(Shader::TypeTask, "main.shader", "TASK_SHADER=1; GROUP_SIZE=%uu; MAX_GEOMETRIES=%u", task_group_size, max_geometries)) return 1;
	if(!pipeline.loadShaderGLSL(Shader::TypeMesh, "main.shader", "MESH_SHADER=1; GROUP_SIZE=%uu; MAX_GEOMETRIES=%u; MAX_VERTICES=%uu; MAX_PRIMITIVES=%uu", mesh_group_size, max_geometries, max_attributes, max_primitives)) return 1;
	if(!pipeline.loadShaderGLSL(Shader::TypeFragment, "main.shader", "FRAGMENT_SHADER=1")) return 1;
	if(!pipeline.create()) return 1;
	
	// create mesh
	if(!File::isFile("model.mesh")) {
		
		Async async;
		if(!async.init()) return 1;
		
		// load mesh
		Mesh src_mesh;
		TS_LOG(Message, "Loading Mesh\n");
		if(!src_mesh.load("model.fbx")) return 1;
		
		// remove texture coordinates
		for(MeshGeometry &geometry : src_mesh.getGeometries()) {
			MeshAttribute texcoord_attribute = geometry.getAttribute(MeshAttribute::TypeTexCoord);
			if(texcoord_attribute) geometry.removeAttribute(texcoord_attribute);
		}
		
		// refine mesh
		Mesh subdiv_mesh;
		uint32_t subdiv_steps = 4;
		TS_LOG(Message, "Refining Mesh\n");
		if(!MeshRefine::subdiv(subdiv_mesh, src_mesh, subdiv_steps)) return 1;
		subdiv_mesh.createNormals(true);
		subdiv_mesh.optimizeAttributes();
		subdiv_mesh.optimizeIndices();
		
		// create graph
		Mesh graph_mesh;
		TS_LOG(Message, "Creating Graph\n");
		MeshGraph::create(graph_mesh, subdiv_mesh, max_attributes, max_primitives, nullptr, &async);
		
		// save mesh
		TS_LOG(Message, "Saving Mesh\n");
		graph_mesh.save("model.mesh");
	}
	
	// load mesh
	Mesh mesh;
	if(!mesh.load("model.mesh")) return 1;
	
	// create model
	MeshModel mesh_model;
	if(!mesh_model.create(device, pipeline, mesh, MeshModel::FlagIndices32 | MeshModel::FlagBufferStorage)) return 1;
	Buffer vertex_buffer = mesh_model.getVertexBuffer();
	Buffer index_buffer = mesh_model.getIndexBuffer();
	
	// create instances
	Array<Matrix4x3f> instance_parameters;
	instance_parameters.resize(grid_size * grid_size * grid_height);
	
	// create geometries
	Array<uint32_t> children_indices;
	Array<GeometryParameters> geometry_parameters(mesh.getNumGeometries());
	for(const MeshGeometry &geometry : mesh.getGeometries()) {
		
		uint32_t index = geometry.getIndex();
		GeometryParameters &parameters = geometry_parameters[index];
		
		// bounding box
		const BoundBoxf &bound_box = geometry.getBoundBox();
		parameters.bound_min.set(bound_box.min, 1.0f);
		parameters.bound_max.set(bound_box.max, 1.0f);
		
		// visibility error
		parameters.error = geometry.getVisibilityError();
		
		// parent indices
		parameters.parent_0 = (geometry.getParent0()) ? geometry.getParent0().getIndex() : Maxu32;
		parameters.parent_1 = (geometry.getParent1()) ? geometry.getParent1().getIndex() : Maxu32;
		
		// children indices
		parameters.num_children = geometry.getNumChildren();
		parameters.base_child = children_indices.size();
		for(const MeshGeometry &child : geometry.getChildren()) {
			children_indices.append(child.getIndex());
		}
		
		// vertices
		parameters.num_vertices = min(mesh_model.getNumGeometryVertices(index), max_attributes);
		parameters.base_vertex = mesh_model.getGeometryBaseVertex(index);
		
		// indices
		parameters.num_indices = min(mesh_model.getNumGeometryIndices(index), max_primitives * 3);
		parameters.base_index = mesh_model.getGeometryBaseIndex(index);
	}
	
	// create instances buffer
	Buffer instances_buffer = device.createBuffer(Buffer::FlagStorage, instance_parameters.bytes());
	if(!instances_buffer) return 1;
	
	// create geometries buffer
	Buffer geometries_buffer = device.createBuffer(Buffer::FlagStorage, geometry_parameters.get(), geometry_parameters.bytes());
	if(!geometries_buffer) return 1;
	
	// create children buffer
	Buffer children_buffer = device.createBuffer(Buffer::FlagStorage, children_indices.get(), children_indices.bytes());
	if(!children_buffer) return 1;
	
	// create target
	Target target = device.createTarget(window);
	target.setClearColor(Color::gray * 0.25f);
	
	// create query
	Query time_query;
	if(device.hasQuery(Query::TypeTime)) time_query = device.createQuery(Query::TypeTime);
	
	// create panel
	Panel panel(device);
	
	// main loop
	window.run([&]() -> bool {
		
		Window::update();
		
		// render window
		if(!window.render()) return false;
		
		// get query
		if(time_query && time_query.isAvailable()) panel.setInfo("\nDraw: " + String::fromTime(time_query.getTime()));
		
		// update panel
		panel.update(window, device, target);
		
		// current time
		float32_t time = (float32_t)Time::seconds();
		
		// time control
		if(window.getKeyboardKey('3')) time = 0.0f;
		if(window.getKeyboardKey('4')) time = Pi / 0.2f;
		
		// transform instances
		instance_parameters.clear();
		float32_t offset = (grid_size - 1.0f) * 0.5f;
		for(uint32_t z = 0; z < grid_height; z++) {
			Matrix4x3f rotate = Matrix4x3f::rotateZ(time * (z * 16.0f - 8.0f)) * Matrix4x3f::rotateY(90.0f);
			for(uint32_t y = 0; y < grid_size; y++) {
				for(uint32_t x = 0; x < grid_size; x++) {
					Matrix4x3f translate = Matrix4x3f::translate((x - offset) * 4.0f, (y - offset) * 4.0f, z * 2.0f);
					instance_parameters.append(translate * rotate);
				}
			}
		}
		
		// update instances buffer
		if(!device.setBuffer(instances_buffer, instance_parameters.get(), instance_parameters.bytes())) return false;
		
		// flush instances buffer
		device.flushBuffer(instances_buffer);
		
		// window target
		target.begin();
		{
			// create command list
			Command command = device.createCommand(target);
			
			// set pipeline
			command.setPipeline(pipeline);
			
			// common parameters
			CommonParameters common_parameters;
			float32_t offset = 1.0f - Tellusim::cos(time * 0.2f);
			common_parameters.camera = Matrix4x4f::rotateZ(time * 2.0f) * Vector4f(Vector3f(32.0f + offset * 24.0f, 0.0f, 8.0f + offset * 8.0f), 1.0f);
			common_parameters.projection = Matrix4x4f::perspective(60.0f, (float32_t)window.getWidth() / window.getHeight(), 0.1f, 1000.0f);
			common_parameters.modelview = Matrix4x4f::lookAt(common_parameters.camera.xyz, Vector3f(0.0f, 0.0f, -16.0f), Vector3f(0.0f, 0.0f, 1.0f));
			if(target.isFlipped()) common_parameters.projection = Matrix4x4f::scale(1.0f, -1.0f, 1.0f) * common_parameters.projection;
			common_parameters.grid_size = grid_size;
			common_parameters.projection_scale = Tellusim::abs(common_parameters.projection.m11) * target.getHeight() * (1.0f / 4.0f);
			common_parameters.target_width = (float32_t)target.getWidth();
			common_parameters.target_height = (float32_t)target.getHeight();
			common_parameters.time = time;
			
			// clip planes
			BoundFrustumf bound_frustum(common_parameters.projection, common_parameters.modelview);
			for(uint32_t i = 0; i < 4; i++) {
				common_parameters.planes[i] = bound_frustum.planes[i];
				common_parameters.signs[i].x = (float32_t)bound_frustum.signs[i][0];
				common_parameters.signs[i].y = (float32_t)bound_frustum.signs[i][1];
				common_parameters.signs[i].z = (float32_t)bound_frustum.signs[i][2];
			}
			
			// lod control
			if(window.getKeyboardKey('1')) common_parameters.projection_scale = 0.0f;
			if(window.getKeyboardKey('2')) common_parameters.projection_scale = 1e6f;
			
			// set common parameters
			command.setUniform(0, common_parameters);
			
			// set storage buffers
			command.setStorageBuffers(0, {
				instances_buffer,
				geometries_buffer,
				children_buffer,
				vertex_buffer,
				index_buffer,
			});
			
			// begin query
			if(time_query) command.beginQuery(time_query);
				
				// draw instances
				command.drawMesh(grid_size, grid_size, grid_height);
				
			// end query
			if(time_query) command.endQuery(time_query);
			
			// draw panel
			panel.draw(command, target);
		}
		target.end();
		
		// present window
		if(!window.present()) return false;
		
		// check errors
		if(!device.check()) return false;
		
		return true;
	});
	
	// finish context
	window.finish();
	
	return 0;
}

/*
 */
#if _WINAPP
	#include <system/TellusimWinApp.h>
	TS_DECLARE_WINAPP_MAIN
#endif
#if _ANDROID
	#include <system/TellusimAndroid.h>
	TS_DECLARE_ANDROID_NATIVE_ACTIVITY
#endif
