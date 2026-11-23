import pygame

W, H = 900, 500
COLORS = {0: (76, 175, 80), 1: (255, 193, 7), 2: (255, 120, 0), 3: (220, 40, 40)}
LABELS = {0: "Well", 1: "Mild", 2: "Moderate", 3: "Crisis"}


def action_name(a):
    if a is None:
        return "Start"
    if a < 5:
        return f"Visit & screen youth {a + 1}"
    if a < 10:
        return f"Refer youth {a - 4} to health centre"
    return "Group session at school"


class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption("CHW Triage & Referral - Kigali")
        self.font = pygame.font.SysFont("arial", 16)
        self.big = pygame.font.SysFont("arial", 22, bold=True)
        self.clock = pygame.time.Clock()
        self.houses = [(80 + i * 160, 260) for i in range(5)]

    def text(self, msg, x, y, font=None, color=(0, 0, 0)):
        self.screen.blit((font or self.font).render(msg, True, color), (x, y))

    def draw(self, env):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        s = self.screen
        s.fill((205, 230, 205))

        # header
        self.text(f"Week {env.week}/20", 360, 10, self.big)
        self.text(f"Action: {action_name(env.last_action)}", 300, 45)

        # school (left) and health centre (right)
        pygame.draw.rect(s, (170, 200, 255), (50, 80, 200, 80))
        self.text("School", 120, 90, self.big)
        pygame.draw.rect(s, (255, 255, 255), (650, 80, 200, 80))
        self.text("Health Centre", 685, 90, self.big)
        self.text(f"Slots used: {int(env.referred.sum())}/2", 690, 125)

        # houses of the 5 youth (colour = true hidden risk, only the viewer sees it)
        for i, (x, y) in enumerate(self.houses):
            risk = int(env.risk[i])
            pygame.draw.polygon(s, (120, 60, 30), [(x - 10, y), (x + 60, y - 40), (x + 130, y)])
            pygame.draw.rect(s, COLORS[risk], (x, y, 120, 90))
            self.text(f"Youth {i + 1}", x + 30, y + 10, self.big)
            self.text(LABELS[risk], x + 30, y + 50)
            self.text(f"Last score: {int(env.last_score[i])}", x, y + 95)
            self.text(f"Weeks unseen: {int(min(env.weeks_since[i], 10))}", x, y + 115)
            if env.referred[i] == 1:
                self.text("REFERRED", x + 20, y + 135, color=(0, 0, 200))

        # where the CHW is this week
        a = env.last_action
        if a is None:
            pos = (450, 180)
        elif a < 5:
            pos = (self.houses[a][0] + 60, self.houses[a][1] - 60)
        elif a < 10:
            pos = (750, 185)
        else:
            pos = (150, 185)
        pygame.draw.circle(s, (30, 60, 200), pos, 16)
        self.text("CHW", pos[0] - 15, pos[1] - 38, color=(30, 60, 200))

        # legend
        for k, (risk, name) in enumerate(LABELS.items()):
            pygame.draw.rect(s, COLORS[risk], (250 + k * 110, 465, 18, 18))
            self.text(name, 275 + k * 110, 465)

        pygame.display.flip()
        self.clock.tick(2)  # 2 frames (weeks) per second

    def save(self, path):
        pygame.image.save(self.screen, path)

    def close(self):
        pygame.quit()