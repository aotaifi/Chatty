import AppKit
import ApplicationServices
import Foundation
func attr(_ e:AXUIElement,_ n:CFString)->AnyObject?{var v:CFTypeRef?; return AXUIElementCopyAttributeValue(e,n,&v) == .success ? v as AnyObject? : nil}
func s(_ e:AXUIElement,_ n:CFString)->String{guard let v=attr(e,n) else{return ""}; return String(describing:v)}
func kids(_ e:AXUIElement)->[AXUIElement]{(attr(e,kAXChildrenAttribute as CFString) as? [AXUIElement]) ?? []}
func idFrom(_ raw:String)->String?{guard let u=URL(string:raw),u.host=="chatgpt.com" else{return nil};let p=u.pathComponents.filter{$0 != "/"};return p.count>=2 && p[0]=="c" ? p[1]:nil}
guard CommandLine.arguments.count==2 else{exit(2)}
let wanted=CommandLine.arguments[1]
guard let app=NSRunningApplication.runningApplications(withBundleIdentifier:"com.openai.codex").first else{exit(3)}
let root=AXUIElementCreateApplication(app.processIdentifier)
var found=false,scanned=0
func walk(_ e:AXUIElement,_ d:Int){if found || d>42 || scanned>30000{return};scanned+=1;if s(e,kAXRoleAttribute as CFString)=="AXWebArea",idFrom(s(e,kAXURLAttribute as CFString))==wanted{found=true;return};for c in kids(e){walk(c,d+1)}}
walk(root,0)
if found{print("READY");exit(0)}
print("NOT_READY");exit(20)
