Walker walker;

void setup() {
  size(500, 500);
  background(255);
  walker = new Walker();
}

void draw() { 
  walker.step();
  walker.display();
}
