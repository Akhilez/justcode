:- dynamic on/2.
:- dynamic clear/1.

clear(table).

exec_state([]).
exec_state([H|T]):-
  assert(H),
  exec_state(T).

move_block(C1, C2):-
  clear(C1),
  clear(C2),
  C1 \== C2.

mov(State):-
  exec_state(State),
  move_block(C1, C2),
  write(C1), nl,
  write(C2), nl,
  create_move(C1, C2, Next),
  write(Next).

create_move(B1, B2, [B1, B2]).

delete_state([Pred|Preds]):-
  retract(Pred),
  delete_state(Preds).
delete_state([]).

?- mov([on(a,table), on(b,a), on(c,b), clear(c)]).
