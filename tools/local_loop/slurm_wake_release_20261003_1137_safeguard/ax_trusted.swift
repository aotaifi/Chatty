import ApplicationServices
if AXIsProcessTrusted() {
    print("TRUSTED=true")
    exit(0)
}
print("TRUSTED=false")
exit(20)
