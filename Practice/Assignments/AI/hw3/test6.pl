check_equivalence([H|[]], Goal):-
  member(H, Goal).
check_equivalence([H|T], Goal):-
  member(H, Goal),
  check_equivalence(T, Goal).

is_found(State, Goal):-
  %State = Goal,
  check_equivalence(State, Goal),
  write("Yes, found"), nl.
is_found(_, _):-
  write("No. Not found."), nl.

?- is_found([on(a,b), on(b,c)], [on(b,c), on(a,b)]).
?- is_found([on(a,b), on(b,c)], [on(a,b), on(b,c)]).
