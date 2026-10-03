import AppKit
import ApplicationServices
import Foundation

func attr(_ e: AXUIElement, _ n: CFString) -> AnyObject? {
    var v: CFTypeRef?
    return AXUIElementCopyAttributeValue(e, n, &v) == .success ? v as AnyObject? : nil
}
func s(_ e: AXUIElement, _ n: CFString) -> String {
    guard let v = attr(e, n) else { return "" }
    return String(describing: v).replacingOccurrences(of: "\n", with: " ")
}
func kids(_ e: AXUIElement) -> [AXUIElement] {
    (attr(e, kAXChildrenAttribute as CFString) as? [AXUIElement]) ?? []
}

guard let app = NSRunningApplication.runningApplications(withBundleIdentifier: "com.openai.codex").first else {
    fputs("ChatGPT not running\n", stderr); exit(2)
}
let root = AXUIElementCreateApplication(app.processIdentifier)
var titles = Set<String>()
var scanned = 0
func walk(_ e: AXUIElement, _ d: Int) {
    if d > 32 || scanned > 30000 { return }
    scanned += 1
    if s(e, kAXRoleAttribute as CFString) == "AXWebArea" {
        let t = s(e, kAXTitleAttribute as CFString).trimmingCharacters(in: .whitespacesAndNewlines)
        if !t.isEmpty && t != "ChatGPT" { titles.insert(t) }
    }
    for c in kids(e) { walk(c, d + 1) }
}
walk(root, 0)

if titles.count == 1, let t = titles.first {
    print(t)
    exit(0)
}
if titles.isEmpty {
    fputs("no titled normal-chat WebArea visible\n", stderr); exit(3)
}
fputs("ambiguous visible WebArea titles: \(titles.sorted())\n", stderr)
exit(4)
