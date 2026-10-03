import AppKit
import ApplicationServices
import Foundation

guard CommandLine.arguments.count >= 3 else {
    fputs("usage: chatty-ax-target get|set|press-send THREAD_ID [TEXT]\n", stderr); exit(2)
}
let command=CommandLine.arguments[1]
let expectedID=CommandLine.arguments[2]
let payload=CommandLine.arguments.count>3 ? CommandLine.arguments.dropFirst(3).joined(separator:" ") : ""
guard let running=NSRunningApplication.runningApplications(withBundleIdentifier:"com.openai.codex").first else{fatalError("ChatGPT not running")}
let app=AXUIElementCreateApplication(running.processIdentifier)
func attr(_ e:AXUIElement,_ n:CFString)->AnyObject?{var v:CFTypeRef?; return AXUIElementCopyAttributeValue(e,n,&v) == .success ? v as AnyObject? : nil}
func str(_ e:AXUIElement,_ n:CFString)->String{guard let v=attr(e,n) else{return ""}; return String(describing:v).replacingOccurrences(of:"\n",with:" ")}
func idFrom(_ raw:String)->String?{guard let u=URL(string:raw),u.host=="chatgpt.com" else{return nil};let p=u.pathComponents.filter{$0 != "/"};return p.count>=2 && p[0]=="c" ? p[1]:nil}
struct Found{var composer:AXUIElement?=nil;var send:AXUIElement?=nil;var stop:AXUIElement?=nil;var web=false;var msg=false}
var found=Found(),targetText="",scanned=0
func walk(_ e:AXUIElement,_ depth:Int,_ inTarget:Bool){
    if depth>42 || scanned>30000{return}; scanned += 1
    let role=str(e,kAXRoleAttribute as CFString)
    let here = role=="AXWebArea" ? (idFrom(str(e,kAXURLAttribute as CFString))==expectedID) : inTarget
    if role=="AXWebArea" && here{found.web=true}
    if here{
        let title=str(e,kAXTitleAttribute as CFString),desc=str(e,kAXDescriptionAttribute as CFString)
        let label=(title+" "+desc).trimmingCharacters(in:.whitespacesAndNewlines).lowercased()
        if role=="AXTextArea" && (title=="Ask ChatGPT" || desc=="Ask ChatGPT" || label.contains("chatgpt")){found.composer=e}
        if role=="AXButton" && (label=="send" || label.hasPrefix("send ") || label.contains("send message") || label.contains("send prompt")){found.send=e}
        if role=="AXButton" && (label=="stop" || label.hasPrefix("stop ")){found.stop=e}
        if role != "AXTextArea"{
            let hay=[title,desc,str(e,kAXValueAttribute as CFString)].joined(separator:" ")
            targetText += " "+hay
            if !payload.isEmpty && hay.contains(payload){found.msg=true}
        }
    }
    if let cs=attr(e,kAXChildrenAttribute as CFString) as? [AXUIElement]{for c in cs{walk(c,depth+1,here)}}
}
guard let wins=attr(app,kAXWindowsAttribute as CFString) as? [AXUIElement] else{fatalError("no windows")}
for w in wins{walk(w,0,false)}
guard found.web else{fputs("target thread web area not found\n",stderr);exit(4)}
if command=="probe"{print("WEB=true");exit(0)}
if command=="contains"{let ok=found.msg || (!payload.isEmpty && targetText.contains(payload));print(ok ? "FOUND=true":"FOUND=false");exit(ok ? 0:9)}
guard let composer=found.composer else{fputs("target composer not found\n",stderr);exit(5)}
if command=="get"{print("VALUE="+str(composer,kAXValueAttribute as CFString));print("SEND="+String(found.send != nil));print("STOP="+String(found.stop != nil));exit(0)}
if command=="set"{
    let current=str(composer,kAXValueAttribute as CFString),trim=current.trimmingCharacters(in:.whitespacesAndNewlines)
    let empty=trim.isEmpty || trim=="Ask ChatGPT"
    if !empty && current != payload{fputs("composer changed before set; refusing overwrite\n",stderr);exit(10)}
    if current==payload{print("SET=already");exit(0)}
    let err=AXUIElementSetAttributeValue(composer,kAXValueAttribute as CFString,payload as CFString);print("SET="+String(err.rawValue));exit(err == .success ? 0:6)
}
if command=="clear-if"{
    guard !payload.isEmpty else{fputs("clear-if requires expected payload\n",stderr);exit(13)}
    let current=str(composer,kAXValueAttribute as CFString)
    guard current==payload else{fputs("composer changed before clear; refusing mutation\n",stderr);exit(14)}
    let err=AXUIElementSetAttributeValue(composer,kAXValueAttribute as CFString,"" as CFString);print("CLEAR="+String(err.rawValue));exit(err == .success ? 0:15)
}
if command=="press-send"{
    guard !payload.isEmpty else{fputs("press-send requires expected payload\n",stderr);exit(11)}
    let current=str(composer,kAXValueAttribute as CFString)
    guard current==payload else{fputs("composer changed before send; refusing submit\n",stderr);exit(12)}
    guard let send=found.send else{fputs("target send button not found\n",stderr);exit(7)}
    let err=AXUIElementPerformAction(send,kAXPressAction as CFString);print("PRESS="+String(err.rawValue));exit(err == .success ? 0:8)
}
fputs("unknown command\n",stderr);exit(3)
