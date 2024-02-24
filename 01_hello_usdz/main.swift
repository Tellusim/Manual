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

import Tellusim

/*
 */
func create_texture(_ device: Device, _ material : MeshMaterial, _ type: String) -> Texture {
	
	// find material parameter
	let index = material.findParameter(type)
	if index == Maxu32 || !material.hasParameterFlag(index, MeshMaterial.Flags.Blob) { return Texture.null() }
	
	// load image
	let image = Image()
	let blob = material.getParameterBlob(index)
	if !image.load(blob) { return Texture.null() }
	
	// create texture
	return device.createTexture(image, Texture.Flags.Mipmaps)
}

/*
 */
func main() -> Int32 {
	
	// create app
	let app = App()
	if !app.create(Platform.Any, UInt32(App.Values.Version.rawValue)) { return 1 }
	
	// create window
	let window = Window(app.getPlatform(), app.getDevice())
	if !window || !window.setSize(app.getWidth(), app.getHeight()) { return 1 }
	if !window.create("01 Hello USDZ") || !window.setHidden(false) { return 1 }
	window.setKeyboardPressedCallback(Window.KeyboardPressedFunction({ (key: UInt32, code: UInt32) in
		if key == Window.Key.Esc.rawValue { window.stop() }
	}))
	
	// create device
	let device = Device(window)
	if !device { return 1 }
	
	// create pipeline
	let pipeline = device.createPipeline()
	pipeline.setSamplerMask(0, Shader.Mask.Fragment)
	pipeline.setTextureMasks(0, 3, Shader.Mask.Fragment)
	pipeline.setUniformMasks(0, 2, Shader.Mask.Vertex)
	pipeline.addAttribute(Pipeline.Attribute.Position, Format.RGBf32, 0, 0, 80)
	pipeline.addAttribute(Pipeline.Attribute.Normal, Format.RGBf32, 0, 12, 80)
	pipeline.addAttribute(Pipeline.Attribute.Tangent, Format.RGBAf32, 0, 24, 80)
	pipeline.addAttribute(Pipeline.Attribute.TexCoord, Format.RGf32, 0, 40, 80)
	pipeline.addAttribute(Pipeline.Attribute.Weights, Format.RGBAf32, 0, 48, 80)
	pipeline.addAttribute(Pipeline.Attribute.Joints, Format.RGBAu32, 0, 64, 80)
	pipeline.setColorFormat(window.getColorFormat())
	pipeline.setDepthFormat(window.getDepthFormat())
	pipeline.setDepthFunc(Pipeline.DepthFunc.Less)
	if !pipeline.loadShaderGLSL(Shader.Kind.Vertex, "main.shader", "VERTEX_SHADER=1") { return 1 }
	if !pipeline.loadShaderGLSL(Shader.Kind.Fragment, "main.shader", "FRAGMENT_SHADER=1") { return 1 }
	if !pipeline.create() { return 1 }
	
	// load mesh
	let mesh = Mesh()
	if !mesh.load("model.usdz") { return 1 }
	if mesh.getNumGeometries() == 0 { return 1 }
	if mesh.getNumAnimations() == 0 { return 1 }
	mesh.setBasis(Mesh.Basis.ZUpRight)
	mesh.createTangents()
	
	// create model
	let model = MeshModel()
	if !model.create(device, pipeline, mesh) { return 1 }
	
	// create textures
	var normal_textures = [Texture]()
	var diffuse_textures = [Texture]()
	var roughness_textures = [Texture]()
	for geometry in mesh.getGeometries() {
		for material in geometry.getMaterials() {
			normal_textures.append(create_texture(device, material, MeshMaterial.TypeNormal))
			diffuse_textures.append(create_texture(device, material, MeshMaterial.TypeDiffuse))
			roughness_textures.append(create_texture(device, material, MeshMaterial.TypeRoughness))
		}
	}
	
	// create sampler
	let sampler = device.createSampler(Sampler.Filter.Trilinear, Sampler.WrapMode.Repeat)
	if !sampler { return 1 }
	
	// create target
	let target = device.createTarget(window)
	target.setClearColor(Color.gray)
	
	// main loop
	window.run(Window.MainLoopFunction({ () -> Bool in
		
		// update window
		Window.update()
		
		// render window
		if !window.render() { return false }
		
		// window target
		if target.begin() {
			
			// create command list
			let command = device.createCommand(target)
			
			// set pipeline
			command.setPipeline(pipeline)
			
			// set sampler
			command.setSampler(0, sampler)
			
			// set model buffers
			model.setBuffers(command)
			
			// set common parameters
			struct CommonParameters {
				var projection: Matrix4x4f
				var modelview: Matrix4x4f
				var camera: Vector4f
			}
			let camera = Vector4f(0.0, -180.0, 180.0, 0.0)
			var common_parameters = CommonParameters(
				projection: Matrix4x4f.perspective(60.0, Float32(window.getWidth()) / Float32(window.getHeight()), 0.1, 1000.0),
				modelview: Matrix4x4f.lookAt(camera.xyz, Vector3f(0.0, 0.0, 80.0), Vector3f(0.0, 0.0, 1.0)),
				camera: camera
			)
			if target.isFlipped() { common_parameters.projection = Matrix4x4f.scale(1.0, -1.0, 1.0) * common_parameters.projection }
			command.setUniform(0, &common_parameters)
			
			// mesh animation
			let time = Time.seconds()
			let animation = mesh.getAnimation(0)
			animation.setTime(time, Matrix4x3d.rotateZ(sin(time) * 30.0))
			
			// draw geometries
			var texture_index = Int(0)
			var joint_parameters = [Vector4f](repeating: Vector4f.zero, count: 192)
			for geometry in mesh.getGeometries() {
				
				// joint transforms
				for i in 0 ..< geometry.getNumJoints() {
					let joint = geometry.getJoint(i)
					let transform = Matrix4x3f(animation.getGlobalTransform(joint)) * joint.getITransform() * geometry.getTransform()
					joint_parameters[Int(i) * 3 + 0] = transform.row_0
					joint_parameters[Int(i) * 3 + 1] = transform.row_1
					joint_parameters[Int(i) * 3 + 2] = transform.row_2
				}
				command.setUniform(1, &joint_parameters)
				
				// draw materials
				for material in geometry.getMaterials() {
					command.setTexture(0, normal_textures[texture_index])
					command.setTexture(1, diffuse_textures[texture_index])
					command.setTexture(2, roughness_textures[texture_index])
					model.draw(command, geometry.getIndex(), material.getIndex())
					texture_index += 1
				}
			}
			
			target.end()
		}
		
		// present window
		if !window.present() { return false }
		
		// check errors
		if !device.check() { return false }
		
		return true
	}))
	
	// finish context
	window.finish()
	
	return 0
}

/*
 */
print(main())
