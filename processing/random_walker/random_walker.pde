Walker walker;

void setup() {
  size(500, 500);
  background(255);
  walker = new MouseWalker();
}

void draw() { 
  walker.step();
  walker.display();
}
