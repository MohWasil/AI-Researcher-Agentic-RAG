class Orchestrator:
    def __init__(self, maker:Maker, checker:Checker, max_iters=2):
        self.maker = maker
        self.checker = checker
        self.max_iters = max_iters

    def handle_query(self, user_query, user_context=None):
        for i in range(self.max_iters):
            candidate = self.maker.make(user_query, user_context or {})
            check = self.checker.run_checks(candidate)
            if check["approved"]:
                return {
                    "answer": candidate["answer"],
                    "provenance": candidate["provenance"],
                    "confidence": check["confidence"]
                }
            else:
                # refine prompt automatically: instruct maker to be conservative & cite sources
                user_query = f"{user_query}\n\nREFINE: Please only assert facts that are directly supported by the provided snippets. If unsupported, say 'I don't know'."
        # after iterations, return candidate with warnings
        return {
            "answer": candidate["answer"],
            "provenance": candidate["provenance"],
            "confidence": check["confidence"],
            "issues": check["issues"]
        }
