class ArchitectureAgent:

    def choose_architecture(self, rows):

        if rows < 5000:

            return [64, 32]

        elif rows < 50000:

            return [128, 64]

        return [256, 128, 64]