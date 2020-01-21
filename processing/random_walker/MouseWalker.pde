class MouseWalker extends Walker {
  @Override
  void step () {
     float choice = random(0, 1); //<>//
     
     float stepX = 0;
     float stepY = 0;
     
     if (choice > 0.8) {
       
       if (mouseX > x) stepX = 1;
       else stepX = -1;
       
       if (mouseY > y) stepY = 1;
       else stepY = -1; 
     
     } else {
        stepX = random(-1, 1);
        stepY = random(-1, 1);
     }
     
     x += stepX;
     y += stepY;
     
  }
}
