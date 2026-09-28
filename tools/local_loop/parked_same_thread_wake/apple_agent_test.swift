import Foundation
import FoundationModels

@main
struct Main {
    static func main() async {
        let model = SystemLanguageModel.default
        print("AVAILABILITY=\(model.availability)")
        guard case .available = model.availability else { return }
        let session = LanguageModelSession()
        do {
            let response = try await session.respond(to: "Reply with exactly: Agent B online")
            print("REPLY=\(response.content)")
        } catch {
            print("ERROR=\(error)")
        }
    }
}
