% Try to create dynamic states for blocks world

:- dynamic on/2.
:- dynamic clear/1.
:- dynamic mov/2.

clear(table).

% ------------RECURSIVE_PREDICATES---------------------

r_put_on(A,B) :-
     on(A,B).
r_put_on(A,B) :-
     not(on(A,B)),
     A \== 'table',
     A \== B,
     clear_off(A),        /* N.B. "action" used as precondition */
     clear_off(B),
     on(A,X),
     retract(on(A,X)),
     assert(on(A,B)),
     retract_clear(B),
     assert(clear(X)),
     assert(move(A,X,B)).

retract_clear(table).
retract_clear(A):-
  A \== 'table',
  retract(clear(A)).

clear_off('table').    /* Means there is room on table */
clear_off(A) :-      /* Means already clear */
     not(on(_X,A)).
clear_off(A) :-
     A \== 'table',
     on(X,A),
     clear_off(X),      /* N.B. recursion */
     retract(on(X,A)),
     assert(on(X,'table')),
     assert(move(X,A,'table')).

% ----------------END RECURSIVE-----------------------

exec_state([]).
exec_state([H|T]):-
  assert(H),
  exec_state(T).

clear_pairs(C1, C2):-
  clear(C1),
  clear(C2),
  not(on(C1, C2)),
  C1 \== C2.

list_state():-
  listing(on),
  listing(clear).

mov(State, Next):-
  write('--------Beginning state'),nl,
  list_state(),

  exec_state(State),
  write('---------Initial state'),nl,
  list_state(),
  
  clear_pairs(B1, B2),
  r_put_on(B1, B2),
  write('-------Moved blocks'),
  write(B1),
  write(B2),nl,
  list_state(),

  current_state(Next),
  write(Next),nl,

  retractall(on(_,_)),
  retractall(clear(_)),
  assert(clear(table)),
  write('-------Cleared state'),nl,
  list_state(),

  fail.
  
  % forall(clear_pairs(C1, C2), write([C1,C2])).

current_state(listing(on)).

delete_state([Pred|Preds]):-
  retract(Pred),
  delete_state(Preds).
delete_state([]).

%-------------Queries--------------------

?- mov([on(a,table), on(c,table), on(b,a), clear(c), clear(b)], _).
