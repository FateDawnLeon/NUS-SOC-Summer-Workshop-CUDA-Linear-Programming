#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include <iostream>
#include <ctime>
#define EPS 1e-5
#define BND 1e5
#define TEST_LINE_LEFT_ID -1
#define TEST_LINE_RIGHT_ID -2
#define TAG_I_PLUS 1
#define TAG_I_MINUS -1
#define TAG_I_ZERO 0

#include "input_output.h"
#include "PerformanceTimer.h"

struct Line
{
	double a, b, c, slope;
	int id, tag;

	__host__ __device__
		Line() { a = 0; b = 0; c = 0; slope = 0; id = -3; tag = -2; }

	__host__ __device__
		Line(double aa, double bb, double cc, int index=-3) {
		a = aa; 
		b = bb; 
		c = cc; 
		id = index;
		if (b > EPS) tag = 1;
		else if (b < -EPS) tag = -1;
		else tag = 0;
		slope = -a / b;
	}

	__host__ __device__
	Line(const Line& L) {
		a = L.a;
		b = L.b;
		c = L.c;
		id = L.id;
		tag = L.tag;
		slope = L.slope;
	}
	
	friend std::ostream &operator<<(std::ostream &stream, const Line &p) {
		stream << p.id << ":(" << p.a << "," << p.b << "," << p.c << ")";
		return stream;
	}
};

struct Point
{
	double x, y;
	int i, j;

	__host__ __device__
		Point() { x = 0; y = 0; i = -3; j = -3; }

	__host__ __device__
		Point(double xx, double yy, int ii, int jj) { x = xx; y = yy; i = ii; j = jj; }

	__host__ __device__
		Point(const Point& p) {
		x = p.x;
		y = p.y;
		i = p.i;
		j = p.j;
	}

	friend std::ostream &operator<<(std::ostream& stream, const Point& p) {
		stream << "[" << p.x << "," << p.y << "]<-(" << p.i << "," << p.j << ")";
		return stream;
	}
};

struct test_line_ip
{
	int cross_line_id, tag;
	double x, y, slope;

	__host__ __device__
		test_line_ip() { cross_line_id = -3; tag = -2; x = 0; y = 0; slope = 0; }

	__host__ __device__
		test_line_ip(int id, int ttag, double xx, double yy, double k) {
		cross_line_id = id;
		tag = ttag;
		x = xx;
		y = yy;
		slope = k;
	}

	__host__ __device__
		test_line_ip(const test_line_ip& p) {
		cross_line_id = p.cross_line_id;
		tag = p.tag;
		x = p.x;
		y = p.y;
		slope = p.slope;
	}

	friend std::ostream &operator<<(std::ostream& stream, const test_line_ip& p) {
		stream << "[" << p.x << "," << p.y << "]:" << p.cross_line_id  << ":" << p.tag << ":" << p.slope;
		return stream;
	}
};

struct compute_test_line_ip
{
	Line test;

	compute_test_line_ip(const Line& l) { test = l; }

	__host__ __device__
		test_line_ip operator() (const Line& line) {
		double a1, a2, b1, b2, c1, c2;
		a1 = test.a;
		b1 = test.b;
		c1 = test.c;
		a2 = line.a;
		b2 = line.b;
		c2 = line.c;
		double x = (c1 * b2 - b1 * c2) / (a1 * b2 - b1 * a2);
		double y = (c1 * a2 - a1 * c2) / (b1 * a2 - a1 * b2);
		return test_line_ip(line.id, line.tag, x, y, line.slope);
	}
};

struct max_iplus
{
	__host__ __device__
		test_line_ip operator() (const test_line_ip& pa, const test_line_ip& pb) {
		if (pa.tag == TAG_I_PLUS && pb.tag == TAG_I_PLUS) {
			return pa.y + EPS < pb.y ? pb : pa;
		}
		else if (pa.tag == TAG_I_PLUS && pb.tag == TAG_I_MINUS) {
			return pa;
		}
		else{
			return pb;
		}
	}
};

struct min_minus
{
	__host__ __device__
		test_line_ip operator() (const test_line_ip& pa, const test_line_ip& pb) {
		if (pa.tag == TAG_I_MINUS && pb.tag == TAG_I_MINUS) {
			return pa.y + EPS < pb.y ? pa : pb;
		}
		else if (pa.tag == TAG_I_MINUS && pb.tag == TAG_I_PLUS) {
			return pa;
		}
		else {
			return pb;
		}
	}
};

struct min_max_pair
{
	test_line_ip max_up;
	test_line_ip min_down;

	min_max_pair() {}

	min_max_pair(test_line_ip mu, test_line_ip md) {
		max_up = mu;
		min_down = md;
	}
};

struct rotate_line
{
	double A, B;

	rotate_line(double obj_a, double obj_b) { A = obj_a; B = obj_b; }

	__host__ __device__
		Line operator() (const Line& line) {
		double a = line.a, b = line.b;

		// rotate the normal vector of line
		double new_a, new_b;
		// new_a = (B,-A) * (a,b) / sqrt(A^2+B^2)
		new_a = (B*a - A*b) / sqrt(A*A + B*B);
		// new_b = (A,B) * (a,b) / sqrt(A^2+B^2)
		new_b = (A*a + B*b) / sqrt(A*A + B*B);
		// rotate the normal vector of line

		return Line(new_a, new_b, line.c, line.id);
	}
};

template<typename T>
void print(const thrust::device_vector<T>& vec) {
	thrust::copy(vec.begin(), vec.end(), std::ostream_iterator<T>(std::cout, "\n"));
	std::cout << std::endl;
}

template <typename T>
void print(const T& obj) {
	std::cout << obj << std::endl;
}

void println() {
	std::cout << std::endl;
}

Point compute_ip(const Line& line1, const Line& line2) {
	double a1, a2, b1, b2, c1, c2;
	a1 = line1.a;
	b1 = line1.b;
	c1 = line1.c;
	a2 = line2.a;
	b2 = line2.b;
	c2 = line2.c;
	double x = (c1 * b2 - b1 * c2) / (a1 * b2 - b1 * a2);
	double y = (c1 * a2 - a1 * c2) / (b1 * a2 - a1 * b2);
	return Point(x, y, line1.id, line2.id);
}

min_max_pair test(double test_x, const thrust::device_vector<Line>& lines, thrust::device_vector<test_line_ip>& test_ips) {
	// calculate all ips with the test line
	Line test_line(1, 0, test_x);
	thrust::transform(lines.begin(), lines.end(), test_ips.begin(), compute_test_line_ip(test_line));
	
	// find the highest I+ line and the lowest I- line
	test_line_ip init_ip = test_ips[0];
	test_line_ip p_up_max = thrust::reduce(test_ips.begin(), test_ips.end(), init_ip, max_iplus());
	test_line_ip p_down_min = thrust::reduce(test_ips.begin(), test_ips.end(), init_ip, min_minus());

	return min_max_pair(p_up_max, p_down_min);
}

min_max_pair find_boundary(double start, const thrust::device_vector<Line>& lines, thrust::device_vector<test_line_ip>& test_ips) {
	double bnd = start;
	min_max_pair mmp;
	Line line1, line2;
	Point ip;
	while (true)
	{
		mmp = test(bnd, lines, test_ips);
		// if there is I- lines and min I- is below the max I+
		if (mmp.min_down.tag == TAG_I_MINUS && mmp.min_down.y + EPS < mmp.max_up.y) {
			// move left boundary to their ip
			line1 = lines[mmp.min_down.cross_line_id];
			line2 = lines[mmp.max_up.cross_line_id];
			ip = compute_ip(line1, line2);
			bnd = ip.x;
		}
		else
		{
			return mmp;
		}
	}
}

void rotate(double* point, double A, double B) {
	double x, y;
	x = point[0];
	y = point[1];

	// rotate the normal vector of line
	double new_a, new_b;
	// new_a = (B,-A) * (a,b) / sqrt(A^2+B^2)
	new_a = (B*x - A*y) / sqrt(A*A + B*B);
	// new_b = (A,B) * (a,b) / sqrt(A^2+B^2)
	new_b = (A*x + B*y) / sqrt(A*A + B*B);

	point[0] = new_a;
	point[1] = new_b;
}

double ans_x, ans_y, left = -BND, right = BND;
test_line_ip left_max_up, right_max_up, tmp_max_up;
min_max_pair left_minmax, right_minmax, mmp;
Line left_line, right_line;
Point next_test_ip;


int main() {
	/* No line removing version */
	// read in all data
	inputs * input = read_from_file("./testcases/100000_0.dat");
	int ans_line_i, ans_line_j, N = input->number;

	// load data from cpu to gpu
	thrust::host_vector<Line> h_lines(N);
	for (int i = 0; i < N; i++) {
		h_lines[i] = Line(input->lines[i]->param_a, input->lines[i]->param_b, input->lines[i]->param_c, i);
	}
	thrust::device_vector<Line> lines = h_lines;
	thrust::device_vector<test_line_ip> test_ips(N);

	// start timeing
	print("Start working on GPU...");
	double start = get_cpu_time();

	// rotate lines
	thrust::transform(lines.begin(), lines.end(), lines.begin(), rotate_line(input->obj_function_param_a, input->obj_function_param_b));

	// first step: narrow down the left and right boundary until both reach the feasiable region
	left_minmax = find_boundary(left, lines, test_ips);
	right_minmax = find_boundary(right, lines, test_ips);
	left_max_up = left_minmax.max_up;
	right_max_up = right_minmax.max_up;

	// second stpe: make sure we do not miss the special cases where the optim point is an ip of a I+ and I-
	if (left_max_up.slope > EPS) {
		ans_x = left_max_up.x;
		ans_y = left_max_up.y;
		ans_line_i = left_max_up.cross_line_id;
		ans_line_j = left_minmax.min_down.cross_line_id;
	}
	else if (right_max_up.slope < -EPS)
	{
		ans_x = right_max_up.x;
		ans_y = right_max_up.y;
		ans_line_i = right_max_up.cross_line_id;
		ans_line_j = right_minmax.min_down.cross_line_id;
	}
	else
	{
		// third step: find new test line between two boundaries and update the L/R boundary until reach the new test exactly hit the optim
		while (true)
		{
			left_line = lines[left_max_up.cross_line_id];
			right_line = lines[right_max_up.cross_line_id];
			next_test_ip = compute_ip(left_line, right_line);
			mmp = test(next_test_ip.x, lines, test_ips);
			tmp_max_up = mmp.max_up;
			if (tmp_max_up.cross_line_id == left_line.id || tmp_max_up.cross_line_id == right_line.id) {
				ans_x = tmp_max_up.x;
				ans_y = tmp_max_up.y;
				ans_line_i = left_line.id;
				ans_line_j = right_line.id;
				print("find answer between the two boundaries.");
				break;
			}
			else
			{
				if (tmp_max_up.slope > EPS) {
					right_max_up = tmp_max_up;
				}
				else if (tmp_max_up.slope < -EPS)
				{
					left_max_up = tmp_max_up;
				}
				else
				{
					print("find answer between the two boundaries and on a horizontal line!");
					print(tmp_max_up.x);
					print(tmp_max_up.y);
					exit(0);
				}
			}
		}
	}

	double end = get_cpu_time();
	print("End working on GPU...");
	
	double A = input->obj_function_param_a;
	double B = input->obj_function_param_b;
	double reverse_A = -A / sqrt(A*A + B*B);
	double reverse_B = B / sqrt(A*A + B*B);

	double ip[2] = { ans_x, ans_y };
	rotate(ip, reverse_A, reverse_B);

	answer * ans = (answer *)malloc(sizeof(answer));
	ans->intersection_point = (point *)malloc(sizeof(point));
	ans->intersection_point->pos_x = ip[0];
	ans->intersection_point->pos_y = ip[1];
	ans->line1 = input->lines[ans_line_i];
	ans->line2 = input->lines[ans_line_j];
	ans->answer_b = A * ip[0] + B * ip[1];

	char * ans_string = generate_ans_string(ans);
	printf("%s", ans_string);
	printf("Time Used:%lf\n", end - start);

	return 0;
}