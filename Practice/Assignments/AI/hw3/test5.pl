% Try to create dynamic states for blocks world

:- dynamic on/2.
:- dynamic clear/1.
:- dynamic mov/2.

clear(t).

% ------------RECURSIVE_PREDICATES---------------------

r_put_on(A,B) :-
     on(A,B).
r_put_on(A,B) :-
     not(on(A,B)),
     A \== t,
     A \== B,
     clear_off(A),        /* N.B. "action" used as precondition */
     clear_off(B),
     on(A,X),
     retract(on(A,X)),
     assert(on(A,B)),
     retract_clear(B),
     assert(clear(X)),
     assert(move(A,X,B)).

retract_clear(t).
retract_clear(A):-
  A \== t,
  retract(clear(A)).

clear_off(t).    /* Means there is room on table */
clear_off(A) :-      /* Means already clear */
     not(on(_X,A)).
clear_off(A) :-
     A \== t,	
     on(X,A),
     clear_off(X),      /* N.B. recursion */
     retract(on(X,A)),
     assert(on(X,t)),
     assert(move(X,A,t)).

% ----------------END RECURSIVE-----------------------

exec_state([]).
exec_state([H|T]):-
  assert(H),
  exec_state(T).

clear_pairs(C1, C2):-
  clear(C1),
  clear(C2),
  not(on(C1, C2)),
  not(C1 = t),
  C1 \== C2.

list_state():-
  listing(on),
  listing(clear).

assert_transition(State, B1, B2):-
  delete(State, on(B1, _), State2),
  delete(State2, clear(B2), State3),
  append(State3, [on(B1,B2)], State4),
  write(State),nl,
  write([B1, B2]),nl,
  write(State4), nl,nl,
  assert(mov(State, State4)).

assert_child_states(State):-
  exec_state(State),
  forall(clear_pairs(C1, C2), assert_transition(State, C1, C2)),
  delete_state(State).

current_state(listing(on)).

delete_state([Pred|Preds]):-
  retract(Pred),
  delete_state(Preds).
delete_state([]).

%-------------Queries--------------------

%?- mov([on(a,t), on(c,t), on(b,a), clear(c), clear(b)], _).

?- listing(mov).

?- assert_child_states([on(a,t), on(c,t), on(b,a), clear(c), clear(b)]).

?- listing(mov).




