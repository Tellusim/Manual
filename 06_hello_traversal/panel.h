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

#ifndef __PANEL_H__
#define __PANEL_H__

#include <core/TellusimTime.h>
#include <interface/TellusimCanvas.h>
#include <interface/TellusimControls.h>

/*
 */
namespace Tellusim {
	
	/*
	 */
	class Command;
	
	/*
	 */
	class Panel {
			
		public:
			
			explicit Panel(const Device &device) {
				
				/// create root
				root = ControlRoot(canvas, true);
				
				/// create panel
				panel = ControlPanel(&root);
				panel.setAlign(Control::AlignLeftTop);
				
				/// create fps text
				fps_text = ControlText(&panel, "FPS: N/A");
				fps_text.setFontSize(22);
				
				/// create info text
				info_text = ControlText(&panel);
				info_text.setEnabled(false);
			}
			~Panel() {
				
			}
			
			/// update panel
			void update(const Window &window, const Device &device, const Target &target) {
				
				/// fps counter
				float64_t current = Time::seconds();
				if(frames++ && current - time > 1.0) {
					fps_text.setText(String::format("FPS: %.1f", frames / (current - time)));
					time = current;
					frames = 0;
				}
				
				/// window size
				float32_t height = 900.0f;
				float32_t width = Tellusim::floor(height * (float32_t)window.getWidth() / (float32_t)window.getHeight());
				
				/// update controls
				root.setViewport(width, height);
				while(root.update(canvas.getScale(target))) { }
				
				/// create canvas
				canvas.create(device, target);
			}
			
			/// draw panel
			void draw(Command &command, const Target &target) {
				canvas.draw(command, target);
			}
			
			/// panel info
			void setInfo(const String &info) {
				info_text.setEnabled((bool)info);
				info_text.setText(info);
			}
			
		private:
			
			Canvas canvas;
			
			ControlRoot root;
			ControlPanel panel;
			ControlText fps_text;
			ControlText info_text;
			
			uint32_t frames = 0;
			float64_t time = 0.0;
	};
};

#endif /* __PANEL_H__ */
