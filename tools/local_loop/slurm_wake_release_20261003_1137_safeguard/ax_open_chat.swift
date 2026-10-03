import AppKit
import ApplicationServices
import CoreGraphics
import Foundation

func val(_ e:AXUIElement,_ n:CFString)->AnyObject? {
    var v:CFTypeRef?
    return AXUIElementCopyAttributeValue(e,n,&v) == .success ? v as AnyObject? : nil
}
func str(_ e:AXUIElement,_ n:CFString)->String {
    guard let v=val(e,n) else{return ""}
    return String(describing:v).replacingOccurrences(of:"\n",with:" ")
}
func role(_ e:AXUIElement)->String { str(e,kAXRoleAttribute as CFString) }
func point(_ e:AXUIElement)->CGPoint? {
    guard let v=val(e,kAXPositionAttribute as CFString) else{return nil}
    var p=CGPoint.zero
    return AXValueGetValue(v as! AXValue,.cgPoint,&p) ? p:nil
}
func size(_ e:AXUIElement)->CGSize? {
    guard let v=val(e,kAXSizeAttribute as CFString) else{return nil}
    var z=CGSize.zero
    return AXValueGetValue(v as! AXValue,.cgSize,&z) ? z:nil
}
func frame(_ e:AXUIElement)->CGRect? {
    guard let p=point(e),let z=size(e) else{return nil}
    return CGRect(origin:p,size:z)
}
func click(_ p:CGPoint) {
    let s=CGEventSource(stateID:.hidSystemState)
    CGEvent(mouseEventSource:s,mouseType:.mouseMoved,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
    usleep(35_000)
    CGEvent(mouseEventSource:s,mouseType:.leftMouseDown,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
    usleep(35_000)
    CGEvent(mouseEventSource:s,mouseType:.leftMouseUp,mouseCursorPosition:p,mouseButton:.left)?.post(tap:.cghidEventTap)
}
func sendKey(_ code:CGKeyCode,_ flags:CGEventFlags=[]) {
    let s=CGEventSource(stateID:.hidSystemState)
    let d=CGEvent(keyboardEventSource:s,virtualKey:code,keyDown:true)!
    let u=CGEvent(keyboardEventSource:s,virtualKey:code,keyDown:false)!
    d.flags=flags;u.flags=flags;d.post(tap:.cghidEventTap);usleep(20_000);u.post(tap:.cghidEventTap)
}
func paste(_ text:String) {
    let pb=NSPasteboard.general, old=pb.string(forType:.string)
    pb.clearContents();pb.setString(text,forType:.string)
    sendKey(0,.maskCommand);usleep(50_000);sendKey(9,.maskCommand);usleep(180_000)
    pb.clearContents();if let old{pb.setString(old,forType:.string)}
}
func windowRect()->CGRect? {
    guard let xs=CGWindowListCopyWindowInfo([.optionOnScreenOnly],kCGNullWindowID) as? [[String:Any]] else{return nil}
    var best:CGRect?
    for w in xs {
        guard (w[kCGWindowOwnerName as String] as? String)=="ChatGPT",
              (w[kCGWindowLayer as String] as? Int)==0,
              let b=w[kCGWindowBounds as String] as? [String:Any],
              let x=b["X"] as? CGFloat,let y=b["Y"] as? CGFloat,
              let ww=b["Width"] as? CGFloat,let hh=b["Height"] as? CGFloat else{continue}
        let r=CGRect(x:x,y:y,width:ww,height:hh)
        if best==nil || r.width*r.height > best!.width*best!.height {best=r}
    }
    return best
}
func rootAt(_ p:CGPoint)->AXUIElement? {
    let sys=AXUIElementCreateSystemWide();var hit:AXUIElement?
    guard AXUIElementCopyElementAtPosition(sys,Float(p.x),Float(p.y),&hit) == .success,var e=hit else{return nil}
    for _ in 0..<20 {
        if role(e)=="AXWebArea"{return e}
        guard let p=val(e,kAXParentAttribute as CFString) else{break}
        e=p as! AXUIElement
    }
    return e
}
func all(_ root:AXUIElement,limit:Int=15000)->[AXUIElement] {
    var out:[AXUIElement]=[],q=[root],n=0
    while !q.isEmpty && n<limit {
        let e=q.removeFirst();out.append(e);n+=1
        if let kids=val(e,kAXChildrenAttribute as CFString) as? [AXUIElement]{q.append(contentsOf:kids)}
    }
    return out
}
func contains(_ e:AXUIElement,_ needle:String)->Bool {
    [str(e,kAXTitleAttribute as CFString),str(e,kAXDescriptionAttribute as CFString),str(e,kAXValueAttribute as CFString)]
        .joined(separator:"\n").localizedCaseInsensitiveContains(needle)
}
func targetWebAreaExists(_ title:String)->Bool {
    guard let app=NSRunningApplication.runningApplications(withBundleIdentifier:"com.openai.codex").first else{return false}
    let ax=AXUIElementCreateApplication(app.processIdentifier)
    guard let wins=val(ax,kAXWindowsAttribute as CFString) as? [AXUIElement] else{return false}
    for w in wins {
        for e in all(w) where role(e)=="AXWebArea" {
            if str(e,kAXTitleAttribute as CFString)==title{return true}
        }
    }
    return false
}

guard CommandLine.arguments.count>=2 else{fputs("usage: chatty-ax-open-chat [--force] TITLE\n",stderr);exit(2)}
let rawArgs=Array(CommandLine.arguments.dropFirst())
let force=rawArgs.contains("--force")
let title=rawArgs.filter{$0 != "--force"}.joined(separator:" ")
guard !title.isEmpty else{fputs("AX_OPEN_EMPTY_TITLE\n",stderr);exit(2)}
if !force && targetWebAreaExists(title){print("AX_OPEN_ALREADY_TARGET");exit(0)}

guard let app=NSRunningApplication.runningApplications(withBundleIdentifier:"com.openai.codex").first else{
    fputs("AX_OPEN_NO_APP\n",stderr);exit(3)
}
app.activate(options:[.activateIgnoringOtherApps])
let appRoot=AXUIElementCreateApplication(app.processIdentifier)

// A minimized/hidden ChatGPT window is still present in the AX tree even when
// CoreGraphics omits it from the on-screen window list. Restore and raise it.
if let wins=val(appRoot,kAXWindowsAttribute as CFString) as? [AXUIElement] {
    for win in wins {
        _=AXUIElementSetAttributeValue(win,kAXMinimizedAttribute as CFString,kCFBooleanFalse)
        _=AXUIElementPerformAction(win,kAXRaiseAction as CFString)
    }
}
usleep(300_000)

func exactSidebarRows()->[(AXUIElement,CGRect)] {
    return all(appRoot,limit:30000).compactMap{e->(AXUIElement,CGRect)? in
        guard role(e)=="AXButton" else{return nil}
        let t=str(e,kAXTitleAttribute as CFString)
        let d=str(e,kAXDescriptionAttribute as CFString)
        guard t==title || d==title, let f=frame(e), f.width>20, f.height>10 else{return nil}
        return(e,f)
    }
}

var rows=exactSidebarRows()
if rows.isEmpty {
    // The sidebar may be collapsed. The native menu item is far more stable
    // than the old Chromium Search control, so use it to reveal the chat list.
    let toggles=all(appRoot,limit:30000).filter{
        role($0)=="AXMenuItem" &&
        str($0,kAXTitleAttribute as CFString)=="Toggle Sidebar"
    }
    if let toggle=toggles.first {
        _=AXUIElementPerformAction(toggle,kAXPressAction as CFString)
        usleep(400_000)
        rows=exactSidebarRows()
    }
}

guard rows.count==1 else {
    if rows.isEmpty {
        fputs("AX_OPEN_NO_EXACT_SIDEBAR_ROW\n",stderr)
    } else {
        fputs("AX_OPEN_AMBIGUOUS_SIDEBAR_ROW\n",stderr)
    }
    exit(4)
}
let result=rows[0]
let press=AXUIElementPerformAction(result.0,kAXPressAction as CFString)
if press != .success { click(CGPoint(x:result.1.midX,y:result.1.midY)) }

for _ in 0..<40 {
    usleep(250_000)
    if targetWebAreaExists(title){print("AX_OPEN_OK \(title)");exit(0)}
}
fputs("AX_OPEN_TARGET_NOT_VISIBLE\n",stderr);exit(9)
