Walker walker;

void setup() {
  walker = new Walker();
}

void draw() { 
  walker.step();
  walker.display();
}
