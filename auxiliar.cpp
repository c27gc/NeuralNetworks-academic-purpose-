#include <iostream>
#include <string.h>
#include <math.h> 

using namespace std;

class col
{
public:
	col()
	{
		a=0;
	};
	col(int k)
	{
		a=k;
	};
	~col();
	float a;
	
};

int main()
{
	col *prueba;
	prueba=new col[3];

	prueba->a=5.5;

	for (int i = 0; i < 3; ++i)
	{
		cout << "Prueba" << prueba[0].a <<endl;
	}
}