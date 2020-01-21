class Walker {
  
  int x;
  int y;
  
  Walker(){
    x = width/2;
    y = height/2;
  }
  
  void step() {
    int choice = int(random(4));
    switch(choice){
      case 0:
        x += 1; break;
      case 1:
        y += 1; break;
      case 2:
        x -= 1; break;
      case 3:
        y -= 1; break;
    }
  }
  
  void display(){
     stroke(0);
     point(x, y);
  }
   
}
