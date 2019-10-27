mov(['a', 'b', 'c'], ['1','2','3']).
mov(['a', 'b', 'c'], ['11','22','33']).
mov(['a', 'b', 'c'], ['111','222','333']).

?- mov(['a', 'b', 'c'], Next), write(Next), nl.

foo([listing(mov)]).

?- foo(X), write(X), nl.

/*
:- dynamic on/2.

mov(State, Next):-
  execute_state(State),
  move_block(C1, C2),
  put_on(C1, C2),
  Next = listing(on),
  delete_state(State).

move_block(C1, C2):-
  clear(C1),
  clear(C2),
  C1 /= C2.

execute_state([Pred|Preds]):-
  call(Pred),
  execute_state(Preds).
execute_state([]).

delete_state([Pred|Preds]):-
  retract(Pred),
  delete_state(Preds).
delete_state([]).


% -------------NON_RECURSIVE_PREDICATES------------------

clear(table).
clear(B) :-
     not(on(_X,B)).

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
     assert(move(A,X,B)).

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

%-------------------------BFS------------------------------

state_record(State, Parent, [State, Parent]).

go(Start, Goal) :-
    empty_queue(Empty_open),
    state_record(Start, nil, State),
    add_to_queue(State, Empty_open, Open),
    empty_set(Closed),
    path(Open, Closed, Goal).

path(Open,_,_) :- empty_queue(Open),
                  write('graph searched, no solution found').

path(Open, Closed, Goal) :-
    remove_from_queue(Next_record, Open, _),
    state_record(State, _, Next_record),
    State = Goal,
    write('Solution path is: '), nl,
    printsolution(Next_record, Closed).

path(Open, Closed, Goal) :-
    remove_from_queue(Next_record, Open, Rest_of_open),
    (bagof(Child, moves(Next_record, Open, Closed, Child), Children);Children = []),
    add_list_to_queue(Children, Rest_of_open, New_open),
    add_to_set(Next_record, Closed, New_closed),
    path(New_open, New_closed, Goal),!.

moves(State_record, Open, Closed, Child_record) :-
    state_record(State, _, State_record),
    mov(State, Next),
    % not (unsafe(Next)),
    state_record(Next, _, Test),
    not(member_queue(Test, Open)),
    not(member_set(Test, Closed)),
    state_record(Next, State, Child_record).

printsolution(State_record, _):-
    state_record(State,nil, State_record),
    write(State), nl.
printsolution(State_record, Closed) :-
    state_record(State, Parent, State_record),
    state_record(Parent, _, Parent_record),
    member(Parent_record, Closed),
    printsolution(Parent_record, Closed),
    write(State), nl.

add_list_to_queue([], Queue, Queue).
add_list_to_queue([H|T], Queue, New_queue) :-
    add_to_queue(H, Queue, Temp_queue),
    add_list_to_queue(T, Temp_queue, New_queue).


%-----------------------QUEUE--------------------------


    % These predicates give a simple,
    % list based implementation of sets

    % empty_set tests/generates an empty set.

empty_set([]).

member_set(E, S) :- member(E, S).

    % add_to_set adds a new member to a set, allowing each element
    % to appear only once

add_to_set(X, S, S) :- member(X, S), !.
add_to_set(X, S, [X|S]).

remove_from_set(_, [], []).
remove_from_set(E, [E|T], T) :- !.
remove_from_set(E, [H|T], [H|T_new]) :-
    remove_from_set(E, T, T_new), !.

union([], S, S).
union([H|T], S, S_new) :-
    union(T, S, S2),
    add_to_set(H, S2, S_new).

intersection([], _, []).
intersection([H|T], S, [H|S_new]) :-
    member_set(H, S),
    intersection(T, S, S_new),!.
intersection([_|T], S, S_new) :-
    intersection(T, S, S_new),!.

set_diff([], _, []).
set_diff([H|T], S, T_new) :-
    member_set(H, S),
    set_diff(T, S, T_new),!.
set_diff([H|T], S, [H|T_new]) :-
    set_diff(T, S, T_new), !.

subset([], _).
subset([H|T], S) :-
    member_set(H, S),
    subset(T, S).

equal_set(S1, S2) :-
    subset(S1, S2), subset(S2, S1).

%%%%%%%%%%%%%%%%%%%%%%% priority queue operations %%%%%%%%%%%%%%%%%%%

    % These predicates provide a simple list based implementation
    % of a priority queue.

    % They assume a definition of precedes for the objects being handled

empty_sort_queue([]).

member_sort_queue(E, S) :- member(E, S).

insert_sort_queue(State, [], [State]).
insert_sort_queue(State, [H | T], [State, H | T]) :-
    precedes(State, H).
insert_sort_queue(State, [H|T], [H | T_new]) :-
    insert_sort_queue(State, T, T_new).

remove_sort_queue(First, [First|Rest], Rest).

% --------------------END QUEUE----------------------------

*/