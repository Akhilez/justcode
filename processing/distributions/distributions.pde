UniformRandom uniformRandom;

void setup() {
  size(640, 240);
  uniformRandom = new UniformRandom();
  uniformRandom.setup();
}

void draw() {
  uniformRandom.draw();
  rect(50, 50, 50, 50);
}
