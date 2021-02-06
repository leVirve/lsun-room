% Definition of room layout

type(1).typeid = 0;
type(1).region{1} = [4 3 5 6];
type(1).region{2} = [3 1 7 5];
type(1).region{3} = [6 5 7 8];
type(1).region{4} = [2 1 3 4];
type(1).region{5} = [8 7 1 2];
type(1).cornermap = [1 2 3 4 5 6 7 8];
type(1).lines = [1 2;3 4;5 6;7 8;1 3;3 5;5 7;7 1];

type(2).typeid = 1;
type(2).region{1} = [3 1 4 6];
type(2).region{2} = [5 4 1 2];
type(2).region{3} = [6 4 5];
type(2).region{4} = [2 1 3];
type(2).region{5} = [];
type(2).cornermap = [3 1 4 5 7 6];
type(2).lines = [1 2;1 3;1 4;4 5;4 6];

type(3).typeid = 2;
type(3).region{1} = [];
type(3).region{2} = [3 1 4 6];
type(3).region{3} = [6 4 5];
type(3).region{4} = [2 1 3];
type(3).region{5} = [5 4 1 2];
type(3).cornermap = [1 2 3 7 8 5];
type(3).lines = [1 2;1 3;1 4;4 5;4 6];

type(4).typeid = 3;
type(4).region{1} = [];
type(4).region{2} = [2 1 3];
type(4).region{3} = [3 1 4];
type(4).region{4} = [];
type(4).region{5} = [4 1 2];
type(4).cornermap = [7 1 5 8];
type(4).lines = [1 2;1 3;1 4];

type(5).typeid = 4;
type(5).region{1} = [2 1 4];
type(5).region{2} = [3 1 2];
type(5).region{3} = [4 1 3];
type(5).region{4} = [];
type(5).region{5} = [];
type(5).cornermap = [5 3 7 6];
type(5).lines = [1 2;1 3;1 4];

type(6).typeid = 5;
type(6).region{1} = [5 4 6];
type(6).region{2} = [2 1 4 5];
type(6).region{3} = [6 4 1 3];
type(6).region{4} = [];
type(6).region{5} = [3 1 2];
type(6).cornermap = [7 1 8 5 3 6];
type(6).lines = [1 2;1 3;1 4;4 5;4 6];

type(7).typeid = 6;
type(7).region{1} = [3 4];
type(7).region{2} = [3 1 2 4];
type(7).region{3} = [];
type(7).region{4} = [];
type(7).region{5} = [2 1];
type(7).cornermap = [1 7 3 5];
type(7).lines = [1 2;3 4];

type(8).typeid = 7;
type(8).region{1} = [];
type(8).region{2} = [2 1 3 4];
type(8).region{3} = [4 3];
type(8).region{4} = [1 2];
type(8).region{5} = [];
type(8).lines = [1 2;3 4];

type(9).typeid = 8;
type(9).region{1} = [];
type(9).region{2} = [1 2];
type(9).region{3} = [];
type(9).region{4} = [];
type(9).region{5} = [2 1];
type(9).cornermap = [1 7];
type(9).lines = [1 2];

type(10).typeid = 9;
type(10).region{1} = [1 2];
type(10).region{2} = [2 1];
type(10).region{3} = [];
type(10).region{4} = [];
type(10).region{5} = [];
type(10).cornermap = [3 5];
type(10).lines = [1 2];

type(11).typeid = 10;
type(11).region{1} = [];
type(11).region{2} = [1 2];
type(11).region{3} = [2 1];
type(11).region{4} = [];
type(11).region{5} = [];
type(11).cornermap = [7 5];
type(11).lines = [1 2];