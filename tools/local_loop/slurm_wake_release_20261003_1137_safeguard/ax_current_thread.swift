import AppKit
import ApplicationServices
import Foundation

func attr(_ e:AXUIElement,_ n:CFString)->AnyObject?{ var v:CFTypeRef?; return AXUIElementCopyAttributeValue(e,n,&v) == .success ? v as AnyObject? : nil }
func s(_ e:AXUIElement,_ n:CFString)->String{ guard let v=attr(e,n) else{return ""}; return String(describing:v).replacingOccurrences(of:"\n",with:" ") }
func kids(_ e:AXUIElement)->[AXUIElement]{ (attr(e,kAXChildrenAttribute as CFString) as? [AXUIElement]) ?? [] }
func threadID(_ raw:String)->String? {
    guard let u=URL(string:raw), u.host=="chatgpt.com" else{return nil}
    let p=u.pathComponents.filter{$0 != "/"}
    guard p.count>=2, p[0]=="c", !p[1].isEmpty else{return nil}
    return p[1]
}
guard let app=NSRunningApplication.runningApplications(withBundleIdentifier:"com.openai.codex").first else{ fputs("ChatGPT not running\n",stderr); exit(2) }
let root=AXUIElementCreateApplication(app.processIdentifier)
var hits:[(String,String)]=[]
var scanned=0
func walk(_ e:AXUIElement,_ d:Int){
    if d>42 || scanned>30000{return}; scanned += 1
    if s(e,kAXRoleAttribute as CFString)=="AXWebArea" {
        let url=s(e,kAXURLAttribute as CFString)
        if let tid=threadID(url) {
            let title=s(e,kAXTitleAttribute as CFString).trimmingCharacters(in:.whitespacesAndNewlines)
            hits.append((tid,title))
        }
    }
    for c in kids(e){ walk(c,d+1) }
}
walk(root,0)
let unique=Dictionary(grouping:hits,by:{$0.0}).mapValues{$0[0].1}
guard unique.count==1, let pair=unique.first else {
    fputs(unique.isEmpty ? "no normal ChatGPT conversation WebArea visible\n" : "ambiguous visible normal ChatGPT WebAreas\n",stderr)
    exit(unique.isEmpty ? 3 : 4)
}
print("THREAD_ID="+pair.key)
print("TITLE="+pair.value)
