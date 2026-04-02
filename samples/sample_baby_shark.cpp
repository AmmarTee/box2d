// SPDX-FileCopyrightText: 2024 Erin Catto
// SPDX-License-Identifier: MIT

// Baby Shark Ocean — Cooperative shark pack hunting with emergent fish schooling
// Inspired by the most-watched YouTube video of all time (16.7B views)

#include "draw.h"
#include "random.h"
#include "sample.h"

#include "box2d/box2d.h"
#include "box2d/math_functions.h"

#include <GLFW/glfw3.h>
#include <imgui.h>
#include <implot.h>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

// --- Neural Network (shared with ecosystem) ---

struct SharkNet
{
	std::vector<float> weights;
	int inputSize;
	int hiddenSize;
	int outputSize;

	void Init( int inSize, int hSize, int outSize )
	{
		inputSize = inSize;
		hiddenSize = hSize;
		outputSize = outSize;
		int total = ( inSize * hSize ) + hSize + ( hSize * outSize ) + outSize;
		weights.resize( total );
		Randomize();
	}

	void Randomize()
	{
		for ( size_t i = 0; i < weights.size(); ++i )
			weights[i] = RandomFloat() * 2.0f;
	}

	void ForwardStore( const float* input, float* output, float* hiddenOut ) const
	{
		int idx = 0;
		for ( int h = 0; h < hiddenSize; ++h )
		{
			float sum = 0.0f;
			for ( int i = 0; i < inputSize; ++i )
				sum += input[i] * weights[idx++];
			sum += weights[idx++];
			hiddenOut[h] = tanhf( sum );
		}
		for ( int o = 0; o < outputSize; ++o )
		{
			float sum = 0.0f;
			for ( int h = 0; h < hiddenSize; ++h )
				sum += hiddenOut[h] * weights[idx++];
			sum += weights[idx++];
			output[o] = tanhf( sum );
		}
	}

	void Mutate( float rate, float strength )
	{
		for ( size_t i = 0; i < weights.size(); ++i )
		{
			float r = (float)RandomInt() / (float)RAND_LIMIT;
			if ( r < rate )
				weights[i] += RandomFloat() * strength;
		}
	}

	void CopyFrom( const SharkNet& other )
	{
		weights = other.weights;
		inputSize = other.inputSize;
		hiddenSize = other.hiddenSize;
		outputSize = other.outputSize;
	}
};

// --- Types ---

enum SharkType
{
	e_babySark = 0,
	e_mamaSark = 1,
	e_daddySark = 2,
};

struct SharkAgent
{
	b2BodyId bodyId;
	b2ShapeId shapeId;
	SharkNet brain;
	SharkType sharkType;
	float energy;
	float heading;
	float age;
	bool alive;
	float fitness;
	int fishCaught;

	// Activations for visualization
	static constexpr int MAX_INPUTS = 80;
	static constexpr int MAX_HIDDEN = 20;
	static constexpr int MAX_OUTPUTS = 4;
	float lastInputs[MAX_INPUTS];
	float lastHidden[MAX_HIDDEN];
	float lastOutputs[MAX_OUTPUTS];
};

struct Fish
{
	b2BodyId bodyId;
	b2ShapeId shapeId;
	SharkNet brain;
	float heading;
	float energy;
	bool alive;
	float fitness; // survival time
	float age;

	static constexpr int MAX_INPUTS = 50;
	static constexpr int MAX_HIDDEN = 12;
	static constexpr int MAX_OUTPUTS = 4;
	float lastInputs[MAX_INPUTS];
	float lastHidden[MAX_HIDDEN];
	float lastOutputs[MAX_OUTPUTS];
};

// --- Constants ---

// Shark sizes: baby < mama < daddy
static constexpr float BABY_RADIUS = 0.35f;
static constexpr float MAMA_RADIUS = 0.5f;
static constexpr float DADDY_RADIUS = 0.7f;

// Shark speeds: baby fastest, daddy slowest
static constexpr float BABY_MAX_SPEED = 9.0f;
static constexpr float MAMA_MAX_SPEED = 7.0f;
static constexpr float DADDY_MAX_SPEED = 5.5f;

// Shark vision
static constexpr float BABY_VISION = 10.0f;
static constexpr float MAMA_VISION = 14.0f;
static constexpr float DADDY_VISION = 18.0f;

static constexpr float FISH_RADIUS = 0.25f;
static constexpr float FISH_MAX_SPEED = 7.5f;
static constexpr float FISH_VISION = 10.0f;

static constexpr float OCEAN_HALF = 35.0f;

// Shark neural net: 8 rays * 4 channels (dist, isFish, isShark, isWall) + energy + speed + 2 nearest ally direction = 36 inputs
// Outputs: turn, speed
static constexpr int SHARK_RAYS = 8;
static constexpr int SHARK_RAY_CHANNELS = 4;
static constexpr int SHARK_NN_INPUTS = SHARK_RAYS * SHARK_RAY_CHANNELS + 4; // +energy, speed, ally_dx, ally_dy
static constexpr int SHARK_NN_HIDDEN = 16;
static constexpr int SHARK_NN_OUTPUTS = 2;

// Fish neural net: 10 rays * 4 channels (dist, isShark, isFish, isWall) + 2 nearest neighbor dir + speed + energy = 44 inputs
// Outputs: turn, speed
static constexpr int FISH_RAYS = 10;
static constexpr int FISH_RAY_CHANNELS = 4;
static constexpr int FISH_NN_INPUTS = FISH_RAYS * FISH_RAY_CHANNELS + 4; // +neighbor_dx, neighbor_dy, speed, energy
static constexpr int FISH_NN_HIDDEN = 12;
static constexpr int FISH_NN_OUTPUTS = 2;

// Category bits
static constexpr uint64_t CAT_WALL = 0x0001;
static constexpr uint64_t CAT_SHARK = 0x0002;
static constexpr uint64_t CAT_FISH = 0x0004;

// Colors
static constexpr b2HexColor COLOR_OCEAN = b2HexColor( 0x0A1929 );
static constexpr b2HexColor COLOR_BABY = b2HexColor( 0x4FC3F7 );  // light blue
static constexpr b2HexColor COLOR_MAMA = b2HexColor( 0xE91E63 );  // pink
static constexpr b2HexColor COLOR_DADDY = b2HexColor( 0x1565C0 ); // dark blue
static constexpr b2HexColor COLOR_FISH = b2HexColor( 0xFFD54F );  // gold/yellow
static constexpr b2HexColor COLOR_FISH_SCHOOL = b2HexColor( 0xFFA726 ); // orange

static float SharkRadius( SharkType t )
{
	switch ( t )
	{
		case e_babySark: return BABY_RADIUS;
		case e_mamaSark: return MAMA_RADIUS;
		case e_daddySark: return DADDY_RADIUS;
	}
	return BABY_RADIUS;
}

static float SharkMaxSpeed( SharkType t )
{
	switch ( t )
	{
		case e_babySark: return BABY_MAX_SPEED;
		case e_mamaSark: return MAMA_MAX_SPEED;
		case e_daddySark: return DADDY_MAX_SPEED;
	}
	return BABY_MAX_SPEED;
}

static float SharkVision( SharkType t )
{
	switch ( t )
	{
		case e_babySark: return BABY_VISION;
		case e_mamaSark: return MAMA_VISION;
		case e_daddySark: return DADDY_VISION;
	}
	return BABY_VISION;
}

static b2HexColor SharkColor( SharkType t )
{
	switch ( t )
	{
		case e_babySark: return COLOR_BABY;
		case e_mamaSark: return COLOR_MAMA;
		case e_daddySark: return COLOR_DADDY;
	}
	return COLOR_BABY;
}

static const char* SharkName( SharkType t )
{
	switch ( t )
	{
		case e_babySark: return "Baby";
		case e_mamaSark: return "Mama";
		case e_daddySark: return "Daddy";
	}
	return "Baby";
}

// User data tag to distinguish sharks from fish in raycasts
struct EntityTag
{
	enum Type { eShark, eFish };
	Type type;
	int index; // index into m_sharks or m_fish
};

class BabySharkOcean : public Sample
{
public:
	explicit BabySharkOcean( SampleContext* context )
		: Sample( context )
	{
		if ( m_context->restart == false )
		{
			m_context->camera.center = { 0.0f, 0.0f };
			m_context->camera.zoom = 25.0f * 3.0f;
		}

		m_babyCount = 3;
		m_mamaCount = 2;
		m_daddyCount = 1;
		m_fishCount = 60;
		m_mutationRate = 0.15f;
		m_mutationStrength = 0.5f;
		m_generation = 0;
		m_generationTimer = 0.0f;
		m_generationDuration = 25.0f;
		m_showRays = false;
		m_selectedShark = -1;
		m_selectedFish = -1;
		m_totalFishCaught = 0;
		m_historyCount = 0;

		CreateOcean();
		SpawnAll();
	}

	~BabySharkOcean() override
	{
		for ( auto& tag : m_tags )
			delete tag;
		m_tags.clear();
	}

	void CreateOcean()
	{
		b2BodyDef bodyDef = b2DefaultBodyDef();
		b2BodyId groundId = b2CreateBody( m_worldId, &bodyDef );

		b2ShapeDef shapeDef = b2DefaultShapeDef();
		shapeDef.filter.categoryBits = CAT_WALL;
		shapeDef.filter.maskBits = CAT_SHARK | CAT_FISH;

		float hs = OCEAN_HALF;
		b2Segment segs[4] = {
			{ { -hs, -hs }, { hs, -hs } },
			{ { hs, -hs }, { hs, hs } },
			{ { hs, hs }, { -hs, hs } },
			{ { -hs, hs }, { -hs, -hs } },
		};
		for ( int i = 0; i < 4; ++i )
			b2CreateSegmentShape( groundId, &shapeDef, &segs[i] );
	}

	EntityTag* MakeTag( EntityTag::Type type, int index )
	{
		EntityTag* tag = new EntityTag{ type, index };
		m_tags.push_back( tag );
		return tag;
	}

	void SpawnAll()
	{
		// Clean up old tags
		for ( auto& tag : m_tags )
			delete tag;
		m_tags.clear();
		m_sharks.clear();
		m_fish.clear();

		int totalSharks = m_babyCount + m_mamaCount + m_daddyCount;
		m_sharks.reserve( totalSharks );
		m_fish.reserve( m_fishCount );

		float hs = OCEAN_HALF - 2.0f;

		// Create sharks
		for ( int i = 0; i < m_babyCount; ++i )
		{
			SharkAgent s = {};
			s.sharkType = e_babySark;
			s.energy = 120.0f;
			s.heading = RandomFloatRange( -B2_PI, B2_PI );
			s.alive = true;
			s.fitness = 0.0f;
			s.fishCaught = 0;
			s.age = 0.0f;
			s.brain.Init( SHARK_NN_INPUTS, SHARK_NN_HIDDEN, SHARK_NN_OUTPUTS );
			memset( s.lastInputs, 0, sizeof( s.lastInputs ) );
			memset( s.lastHidden, 0, sizeof( s.lastHidden ) );
			memset( s.lastOutputs, 0, sizeof( s.lastOutputs ) );
			m_sharks.push_back( s );
		}
		for ( int i = 0; i < m_mamaCount; ++i )
		{
			SharkAgent s = {};
			s.sharkType = e_mamaSark;
			s.energy = 160.0f;
			s.heading = RandomFloatRange( -B2_PI, B2_PI );
			s.alive = true;
			s.fitness = 0.0f;
			s.fishCaught = 0;
			s.age = 0.0f;
			s.brain.Init( SHARK_NN_INPUTS, SHARK_NN_HIDDEN, SHARK_NN_OUTPUTS );
			memset( s.lastInputs, 0, sizeof( s.lastInputs ) );
			memset( s.lastHidden, 0, sizeof( s.lastHidden ) );
			memset( s.lastOutputs, 0, sizeof( s.lastOutputs ) );
			m_sharks.push_back( s );
		}
		for ( int i = 0; i < m_daddyCount; ++i )
		{
			SharkAgent s = {};
			s.sharkType = e_daddySark;
			s.energy = 200.0f;
			s.heading = RandomFloatRange( -B2_PI, B2_PI );
			s.alive = true;
			s.fitness = 0.0f;
			s.fishCaught = 0;
			s.age = 0.0f;
			s.brain.Init( SHARK_NN_INPUTS, SHARK_NN_HIDDEN, SHARK_NN_OUTPUTS );
			memset( s.lastInputs, 0, sizeof( s.lastInputs ) );
			memset( s.lastHidden, 0, sizeof( s.lastHidden ) );
			memset( s.lastOutputs, 0, sizeof( s.lastOutputs ) );
			m_sharks.push_back( s );
		}

		// Create shark bodies (stable pointers after reserve)
		for ( int i = 0; i < (int)m_sharks.size(); ++i )
		{
			SharkAgent& s = m_sharks[i];
			b2Vec2 pos = { RandomFloatRange( -hs, hs ), RandomFloatRange( -hs, hs ) };
			EntityTag* tag = MakeTag( EntityTag::eShark, i );

			b2BodyDef bd = b2DefaultBodyDef();
			bd.type = b2_dynamicBody;
			bd.position = pos;
			bd.linearDamping = 3.0f;
			bd.motionLocks.angularZ = true;
			bd.userData = tag;
			s.bodyId = b2CreateBody( m_worldId, &bd );

			b2ShapeDef sd = b2DefaultShapeDef();
			sd.density = 1.0f;
			sd.filter.categoryBits = CAT_SHARK;
			sd.filter.maskBits = CAT_WALL | CAT_FISH;
			sd.enableContactEvents = true;
			sd.enableSensorEvents = true;

			b2Circle circle = { { 0.0f, 0.0f }, SharkRadius( s.sharkType ) };
			s.shapeId = b2CreateCircleShape( s.bodyId, &sd, &circle );
		}

		// Create fish
		for ( int i = 0; i < m_fishCount; ++i )
		{
			Fish f = {};
			f.heading = RandomFloatRange( -B2_PI, B2_PI );
			f.energy = 80.0f;
			f.alive = true;
			f.fitness = 0.0f;
			f.age = 0.0f;
			f.brain.Init( FISH_NN_INPUTS, FISH_NN_HIDDEN, FISH_NN_OUTPUTS );
			memset( f.lastInputs, 0, sizeof( f.lastInputs ) );
			memset( f.lastHidden, 0, sizeof( f.lastHidden ) );
			memset( f.lastOutputs, 0, sizeof( f.lastOutputs ) );
			m_fish.push_back( f );
		}

		for ( int i = 0; i < (int)m_fish.size(); ++i )
		{
			Fish& f = m_fish[i];
			b2Vec2 pos = { RandomFloatRange( -hs * 0.6f, hs * 0.6f ), RandomFloatRange( -hs * 0.6f, hs * 0.6f ) };
			EntityTag* tag = MakeTag( EntityTag::eFish, i );

			b2BodyDef bd = b2DefaultBodyDef();
			bd.type = b2_dynamicBody;
			bd.position = pos;
			bd.linearDamping = 4.0f;
			bd.motionLocks.angularZ = true;
			bd.userData = tag;
			f.bodyId = b2CreateBody( m_worldId, &bd );

			b2ShapeDef sd = b2DefaultShapeDef();
			sd.density = 0.5f;
			sd.filter.categoryBits = CAT_FISH;
			sd.filter.maskBits = CAT_WALL | CAT_SHARK | CAT_FISH;
			sd.enableContactEvents = true;

			b2Circle circle = { { 0.0f, 0.0f }, FISH_RADIUS };
			f.shapeId = b2CreateCircleShape( f.bodyId, &sd, &circle );
		}
	}

	// Find nearest ally shark position for cooperative sensing
	b2Vec2 NearestAllyDir( int sharkIdx )
	{
		b2Vec2 myPos = b2Body_GetPosition( m_sharks[sharkIdx].bodyId );
		float bestDist = 1e9f;
		b2Vec2 bestDir = { 0.0f, 0.0f };
		for ( int j = 0; j < (int)m_sharks.size(); ++j )
		{
			if ( j == sharkIdx || !m_sharks[j].alive )
				continue;
			b2Vec2 oPos = b2Body_GetPosition( m_sharks[j].bodyId );
			float dx = oPos.x - myPos.x;
			float dy = oPos.y - myPos.y;
			float d = sqrtf( dx * dx + dy * dy );
			if ( d < bestDist )
			{
				bestDist = d;
				float invD = d > 0.01f ? 1.0f / d : 0.0f;
				bestDir = { dx * invD, dy * invD };
			}
		}
		return bestDir;
	}

	// Find nearest fish neighbor direction for schooling input
	b2Vec2 NearestFishNeighborDir( int fishIdx )
	{
		b2Vec2 myPos = b2Body_GetPosition( m_fish[fishIdx].bodyId );
		float bestDist = 1e9f;
		b2Vec2 bestDir = { 0.0f, 0.0f };
		for ( int j = 0; j < (int)m_fish.size(); ++j )
		{
			if ( j == fishIdx || !m_fish[j].alive )
				continue;
			b2Vec2 oPos = b2Body_GetPosition( m_fish[j].bodyId );
			float dx = oPos.x - myPos.x;
			float dy = oPos.y - myPos.y;
			float d = sqrtf( dx * dx + dy * dy );
			if ( d < bestDist && d < FISH_VISION )
			{
				bestDist = d;
				float invD = d > 0.01f ? 1.0f / d : 0.0f;
				bestDir = { dx * invD, dy * invD };
			}
		}
		return bestDir;
	}

	void CastSharkRays( SharkAgent& shark, int sharkIdx, float* inputs, bool drawRays )
	{
		b2Vec2 pos = b2Body_GetPosition( shark.bodyId );
		float vision = SharkVision( shark.sharkType );

		b2QueryFilter filter = b2DefaultQueryFilter();
		filter.categoryBits = CAT_FISH | CAT_WALL | CAT_SHARK;
		filter.maskBits = CAT_FISH | CAT_WALL | CAT_SHARK;

		float angleStep = 2.0f * B2_PI / (float)SHARK_RAYS;

		for ( int r = 0; r < SHARK_RAYS; ++r )
		{
			float angle = shark.heading + ( r - SHARK_RAYS / 2 ) * angleStep;
			b2Vec2 dir = { cosf( angle ), sinf( angle ) };
			b2Vec2 translation = { dir.x * vision, dir.y * vision };

			b2RayResult result = b2World_CastRayClosest( m_worldId, pos, translation, filter );

			int base = r * SHARK_RAY_CHANNELS;
			if ( result.hit )
			{
				inputs[base + 0] = 1.0f - result.fraction;
				inputs[base + 1] = 0.0f; // isFish
				inputs[base + 2] = 0.0f; // isShark
				inputs[base + 3] = 0.0f; // isWall

				b2BodyId hitBody = b2Shape_GetBody( result.shapeId );
				void* ud = b2Body_GetUserData( hitBody );
				if ( ud != nullptr )
				{
					EntityTag* tag = (EntityTag*)ud;
					if ( tag->type == EntityTag::eFish )
						inputs[base + 1] = 1.0f;
					else if ( tag->type == EntityTag::eShark )
						inputs[base + 2] = 1.0f;
				}
				else
				{
					inputs[base + 3] = 1.0f;
				}

				if ( drawRays )
				{
					b2Vec2 hitPt = { pos.x + translation.x * result.fraction, pos.y + translation.y * result.fraction };
					b2HexColor col = b2_colorGray;
					if ( inputs[base + 1] > 0.5f )
						col = COLOR_FISH;
					else if ( inputs[base + 2] > 0.5f )
						col = COLOR_BABY;
					DrawLine( m_draw, pos, hitPt, col );
				}
			}
			else
			{
				inputs[base + 0] = 0.0f;
				inputs[base + 1] = 0.0f;
				inputs[base + 2] = 0.0f;
				inputs[base + 3] = 0.0f;
			}
		}

		// Extra inputs
		int extraBase = SHARK_RAYS * SHARK_RAY_CHANNELS;
		float maxEnergy = 250.0f;
		inputs[extraBase + 0] = shark.energy / maxEnergy;

		b2Vec2 vel = b2Body_GetLinearVelocity( shark.bodyId );
		inputs[extraBase + 1] = b2Length( vel ) / SharkMaxSpeed( shark.sharkType );

		b2Vec2 allyDir = NearestAllyDir( sharkIdx );
		inputs[extraBase + 2] = allyDir.x;
		inputs[extraBase + 3] = allyDir.y;
	}

	void CastFishRays( Fish& fish, int fishIdx, float* inputs, bool drawRays )
	{
		b2Vec2 pos = b2Body_GetPosition( fish.bodyId );

		b2QueryFilter filter = b2DefaultQueryFilter();
		filter.categoryBits = CAT_SHARK | CAT_FISH | CAT_WALL;
		filter.maskBits = CAT_SHARK | CAT_FISH | CAT_WALL;

		float angleStep = 2.0f * B2_PI / (float)FISH_RAYS;

		for ( int r = 0; r < FISH_RAYS; ++r )
		{
			float angle = fish.heading + ( r - FISH_RAYS / 2 ) * angleStep;
			b2Vec2 dir = { cosf( angle ), sinf( angle ) };
			b2Vec2 translation = { dir.x * FISH_VISION, dir.y * FISH_VISION };

			b2RayResult result = b2World_CastRayClosest( m_worldId, pos, translation, filter );

			int base = r * FISH_RAY_CHANNELS;
			if ( result.hit )
			{
				inputs[base + 0] = 1.0f - result.fraction;
				inputs[base + 1] = 0.0f; // isShark
				inputs[base + 2] = 0.0f; // isFish
				inputs[base + 3] = 0.0f; // isWall

				b2BodyId hitBody = b2Shape_GetBody( result.shapeId );
				void* ud = b2Body_GetUserData( hitBody );
				if ( ud != nullptr )
				{
					EntityTag* tag = (EntityTag*)ud;
					if ( tag->type == EntityTag::eShark )
						inputs[base + 1] = 1.0f;
					else if ( tag->type == EntityTag::eFish )
						inputs[base + 2] = 1.0f;
				}
				else
				{
					inputs[base + 3] = 1.0f;
				}

				if ( drawRays )
				{
					b2Vec2 hitPt = { pos.x + translation.x * result.fraction, pos.y + translation.y * result.fraction };
					b2HexColor col = b2_colorGray;
					if ( inputs[base + 1] > 0.5f )
						col = b2_colorRed;
					else if ( inputs[base + 2] > 0.5f )
						col = COLOR_FISH_SCHOOL;
					DrawLine( m_draw, pos, hitPt, col );
				}
			}
			else
			{
				inputs[base + 0] = 0.0f;
				inputs[base + 1] = 0.0f;
				inputs[base + 2] = 0.0f;
				inputs[base + 3] = 0.0f;
			}
		}

		int extraBase = FISH_RAYS * FISH_RAY_CHANNELS;
		b2Vec2 neighborDir = NearestFishNeighborDir( fishIdx );
		inputs[extraBase + 0] = neighborDir.x;
		inputs[extraBase + 1] = neighborDir.y;

		b2Vec2 vel = b2Body_GetLinearVelocity( fish.bodyId );
		inputs[extraBase + 2] = b2Length( vel ) / FISH_MAX_SPEED;
		inputs[extraBase + 3] = fish.energy / 100.0f;
	}

	void UpdateShark( SharkAgent& shark, int idx, float dt )
	{
		if ( !shark.alive )
			return;

		float inputs[SHARK_NN_INPUTS];
		memset( inputs, 0, sizeof( inputs ) );
		bool draw = m_showRays || idx == m_selectedShark;
		CastSharkRays( shark, idx, inputs, draw );

		float outputs[SHARK_NN_OUTPUTS];
		float hidden[SHARK_NN_HIDDEN];
		shark.brain.ForwardStore( inputs, outputs, hidden );

		memcpy( shark.lastInputs, inputs, sizeof( float ) * SHARK_NN_INPUTS );
		memcpy( shark.lastHidden, hidden, sizeof( float ) * SHARK_NN_HIDDEN );
		memcpy( shark.lastOutputs, outputs, sizeof( float ) * SHARK_NN_OUTPUTS );

		shark.heading += outputs[0] * ( B2_PI / 3.0f ) * dt;
		float maxSpd = SharkMaxSpeed( shark.sharkType );
		float spd = ( outputs[1] + 1.0f ) * 0.5f * maxSpd;

		b2Vec2 velocity = { cosf( shark.heading ) * spd, sinf( shark.heading ) * spd };
		b2Body_SetLinearVelocity( shark.bodyId, velocity );

		float moveCost = 0.3f + spd * 0.2f;
		shark.energy -= moveCost * dt;
		shark.age += dt;

		if ( shark.energy <= 0.0f )
		{
			shark.alive = false;
			b2Body_Disable( shark.bodyId );
		}
	}

	void UpdateFish( Fish& fish, int idx, float dt )
	{
		if ( !fish.alive )
			return;

		float inputs[FISH_NN_INPUTS];
		memset( inputs, 0, sizeof( inputs ) );
		bool draw = m_showRays || idx == m_selectedFish;
		CastFishRays( fish, idx, inputs, draw );

		float outputs[FISH_NN_OUTPUTS];
		float hidden[FISH_NN_HIDDEN];
		fish.brain.ForwardStore( inputs, outputs, hidden );

		memcpy( fish.lastInputs, inputs, sizeof( float ) * FISH_NN_INPUTS );
		memcpy( fish.lastHidden, hidden, sizeof( float ) * FISH_NN_HIDDEN );
		memcpy( fish.lastOutputs, outputs, sizeof( float ) * FISH_NN_OUTPUTS );

		fish.heading += outputs[0] * ( B2_PI / 3.0f ) * dt;
		float spd = ( outputs[1] + 1.0f ) * 0.5f * FISH_MAX_SPEED;

		b2Vec2 velocity = { cosf( fish.heading ) * spd, sinf( fish.heading ) * spd };
		b2Body_SetLinearVelocity( fish.bodyId, velocity );

		fish.energy -= 0.2f * dt;
		fish.age += dt;
		fish.fitness += dt; // survival time

		if ( fish.energy <= 0.0f )
		{
			fish.alive = false;
			b2Body_Disable( fish.bodyId );
		}
	}

	void ProcessContacts()
	{
		b2ContactEvents contactEvents = b2World_GetContactEvents( m_worldId );
		for ( int i = 0; i < contactEvents.beginCount; ++i )
		{
			b2ContactBeginTouchEvent event = contactEvents.beginEvents[i];
			b2BodyId bodyA = b2Shape_GetBody( event.shapeIdA );
			b2BodyId bodyB = b2Shape_GetBody( event.shapeIdB );

			void* udA = b2Body_GetUserData( bodyA );
			void* udB = b2Body_GetUserData( bodyB );

			if ( !udA || !udB )
				continue;

			EntityTag* tagA = (EntityTag*)udA;
			EntityTag* tagB = (EntityTag*)udB;

			EntityTag* sharkTag = nullptr;
			EntityTag* fishTag = nullptr;

			if ( tagA->type == EntityTag::eShark && tagB->type == EntityTag::eFish )
			{
				sharkTag = tagA;
				fishTag = tagB;
			}
			else if ( tagB->type == EntityTag::eShark && tagA->type == EntityTag::eFish )
			{
				sharkTag = tagB;
				fishTag = tagA;
			}

			if ( sharkTag && fishTag )
			{
				SharkAgent& shark = m_sharks[sharkTag->index];
				Fish& fish = m_fish[fishTag->index];

				if ( shark.alive && fish.alive )
				{
					fish.alive = false;
					b2Body_Disable( fish.bodyId );

					// Bigger sharks get more energy from catches
					float bonus = 30.0f + SharkRadius( shark.sharkType ) * 40.0f;
					shark.energy += bonus;
					shark.fishCaught++;
					shark.fitness += 40.0f;

					// Team bonus: all alive sharks get a small fitness bump
					for ( auto& s : m_sharks )
					{
						if ( s.alive )
							s.fitness += 5.0f;
					}

					m_totalFishCaught++;
				}
			}
		}
	}

	void Evolve()
	{
		m_generation++;

		// Record history
		if ( m_historyCount < MAX_HISTORY )
		{
			m_histGeneration[m_historyCount] = (float)m_generation;

			// Best shark fitness
			float bestSharkFit = 0.0f;
			for ( auto& s : m_sharks )
				bestSharkFit = std::max( bestSharkFit, s.fitness );
			m_histSharkFitness[m_historyCount] = bestSharkFit;

			// Best fish fitness (survival)
			float bestFishFit = 0.0f;
			int fishAlive = 0;
			for ( auto& f : m_fish )
			{
				bestFishFit = std::max( bestFishFit, f.fitness );
				if ( f.alive )
					fishAlive++;
			}
			m_histFishFitness[m_historyCount] = bestFishFit;
			m_histFishAlive[m_historyCount] = (float)fishAlive;
			m_histFishCaught[m_historyCount] = (float)m_totalFishCaught;
			m_historyCount++;
		}

		// Sort sharks by fitness
		std::vector<int> sharkOrder( m_sharks.size() );
		for ( int i = 0; i < (int)sharkOrder.size(); ++i )
			sharkOrder[i] = i;
		std::sort( sharkOrder.begin(), sharkOrder.end(),
				   [this]( int a, int b ) { return m_sharks[a].fitness > m_sharks[b].fitness; } );

		// Top half brains by shark type
		std::vector<SharkNet> bestBrains[3]; // indexed by SharkType
		for ( int i = 0; i < (int)sharkOrder.size(); ++i )
		{
			SharkAgent& s = m_sharks[sharkOrder[i]];
			int t = (int)s.sharkType;
			int maxElites = 2;
			if ( (int)bestBrains[t].size() < maxElites )
				bestBrains[t].push_back( s.brain );
		}

		// Sort fish by fitness
		std::vector<int> fishOrder( m_fish.size() );
		for ( int i = 0; i < (int)fishOrder.size(); ++i )
			fishOrder[i] = i;
		std::sort( fishOrder.begin(), fishOrder.end(),
				   [this]( int a, int b ) { return m_fish[a].fitness > m_fish[b].fitness; } );

		int fishElite = std::max( 1, (int)m_fish.size() / 4 );
		std::vector<SharkNet> bestFishBrains;
		for ( int i = 0; i < fishElite; ++i )
			bestFishBrains.push_back( m_fish[fishOrder[i]].brain );

		m_bestSharkFitness = sharkOrder.empty() ? 0.0f : m_sharks[sharkOrder[0]].fitness;
		m_bestFishFitness = fishOrder.empty() ? 0.0f : m_fish[fishOrder[0]].fitness;

		// Destroy all bodies
		for ( auto& s : m_sharks )
			if ( b2Body_IsValid( s.bodyId ) )
				b2DestroyBody( s.bodyId );
		for ( auto& f : m_fish )
			if ( b2Body_IsValid( f.bodyId ) )
				b2DestroyBody( f.bodyId );
		for ( auto& tag : m_tags )
			delete tag;
		m_tags.clear();
		m_sharks.clear();
		m_fish.clear();

		// Respawn sharks with evolved brains
		int totalSharks = m_babyCount + m_mamaCount + m_daddyCount;
		m_sharks.reserve( totalSharks );
		m_fish.reserve( m_fishCount );

		auto spawnSharksOfType = [&]( SharkType type, int count, float baseEnergy )
		{
			int t = (int)type;
			for ( int i = 0; i < count; ++i )
			{
				SharkAgent s = {};
				s.sharkType = type;
				s.energy = baseEnergy;
				s.heading = RandomFloatRange( -B2_PI, B2_PI );
				s.alive = true;
				s.fitness = 0.0f;
				s.fishCaught = 0;
				s.age = 0.0f;
				memset( s.lastInputs, 0, sizeof( s.lastInputs ) );
				memset( s.lastHidden, 0, sizeof( s.lastHidden ) );
				memset( s.lastOutputs, 0, sizeof( s.lastOutputs ) );

				if ( !bestBrains[t].empty() )
				{
					s.brain.CopyFrom( bestBrains[t][i % bestBrains[t].size()] );
					s.brain.Mutate( m_mutationRate, m_mutationStrength );
				}
				else
				{
					s.brain.Init( SHARK_NN_INPUTS, SHARK_NN_HIDDEN, SHARK_NN_OUTPUTS );
				}
				m_sharks.push_back( s );
			}
		};

		spawnSharksOfType( e_babySark, m_babyCount, 120.0f );
		spawnSharksOfType( e_mamaSark, m_mamaCount, 160.0f );
		spawnSharksOfType( e_daddySark, m_daddyCount, 200.0f );

		float hs = OCEAN_HALF - 2.0f;
		for ( int i = 0; i < (int)m_sharks.size(); ++i )
		{
			SharkAgent& s = m_sharks[i];
			b2Vec2 pos = { RandomFloatRange( -hs, hs ), RandomFloatRange( -hs, hs ) };
			EntityTag* tag = MakeTag( EntityTag::eShark, i );

			b2BodyDef bd = b2DefaultBodyDef();
			bd.type = b2_dynamicBody;
			bd.position = pos;
			bd.linearDamping = 3.0f;
			bd.motionLocks.angularZ = true;
			bd.userData = tag;
			s.bodyId = b2CreateBody( m_worldId, &bd );

			b2ShapeDef sd = b2DefaultShapeDef();
			sd.density = 1.0f;
			sd.filter.categoryBits = CAT_SHARK;
			sd.filter.maskBits = CAT_WALL | CAT_FISH;
			sd.enableContactEvents = true;
			sd.enableSensorEvents = true;

			b2Circle circle = { { 0.0f, 0.0f }, SharkRadius( s.sharkType ) };
			s.shapeId = b2CreateCircleShape( s.bodyId, &sd, &circle );
		}

		// Respawn fish with evolved brains
		for ( int i = 0; i < m_fishCount; ++i )
		{
			Fish f = {};
			f.heading = RandomFloatRange( -B2_PI, B2_PI );
			f.energy = 80.0f;
			f.alive = true;
			f.fitness = 0.0f;
			f.age = 0.0f;
			memset( f.lastInputs, 0, sizeof( f.lastInputs ) );
			memset( f.lastHidden, 0, sizeof( f.lastHidden ) );
			memset( f.lastOutputs, 0, sizeof( f.lastOutputs ) );

			if ( !bestFishBrains.empty() )
			{
				f.brain.CopyFrom( bestFishBrains[i % bestFishBrains.size()] );
				f.brain.Mutate( m_mutationRate, m_mutationStrength );
			}
			else
			{
				f.brain.Init( FISH_NN_INPUTS, FISH_NN_HIDDEN, FISH_NN_OUTPUTS );
			}
			m_fish.push_back( f );
		}

		for ( int i = 0; i < (int)m_fish.size(); ++i )
		{
			Fish& f = m_fish[i];
			b2Vec2 pos = { RandomFloatRange( -hs * 0.6f, hs * 0.6f ), RandomFloatRange( -hs * 0.6f, hs * 0.6f ) };
			EntityTag* tag = MakeTag( EntityTag::eFish, i );

			b2BodyDef bd = b2DefaultBodyDef();
			bd.type = b2_dynamicBody;
			bd.position = pos;
			bd.linearDamping = 4.0f;
			bd.motionLocks.angularZ = true;
			bd.userData = tag;
			f.bodyId = b2CreateBody( m_worldId, &bd );

			b2ShapeDef sd = b2DefaultShapeDef();
			sd.density = 0.5f;
			sd.filter.categoryBits = CAT_FISH;
			sd.filter.maskBits = CAT_WALL | CAT_SHARK | CAT_FISH;
			sd.enableContactEvents = true;

			b2Circle circle = { { 0.0f, 0.0f }, FISH_RADIUS };
			f.shapeId = b2CreateCircleShape( f.bodyId, &sd, &circle );
		}

		m_generationTimer = 0.0f;
		m_totalFishCaught = 0;
		m_selectedShark = -1;
		m_selectedFish = -1;
	}

	void MouseDown( b2Vec2 p, int button, int mod ) override
	{
		if ( button == GLFW_MOUSE_BUTTON_1 && ( mod & GLFW_MOD_SHIFT ) )
		{
			float bestDist = 3.0f;
			m_selectedShark = -1;
			m_selectedFish = -1;

			for ( int i = 0; i < (int)m_sharks.size(); ++i )
			{
				if ( !m_sharks[i].alive )
					continue;
				b2Vec2 sp = b2Body_GetPosition( m_sharks[i].bodyId );
				float d = sqrtf( ( sp.x - p.x ) * ( sp.x - p.x ) + ( sp.y - p.y ) * ( sp.y - p.y ) );
				if ( d < bestDist )
				{
					bestDist = d;
					m_selectedShark = i;
					m_selectedFish = -1;
				}
			}

			for ( int i = 0; i < (int)m_fish.size(); ++i )
			{
				if ( !m_fish[i].alive )
					continue;
				b2Vec2 fp = b2Body_GetPosition( m_fish[i].bodyId );
				float d = sqrtf( ( fp.x - p.x ) * ( fp.x - p.x ) + ( fp.y - p.y ) * ( fp.y - p.y ) );
				if ( d < bestDist )
				{
					bestDist = d;
					m_selectedFish = i;
					m_selectedShark = -1;
				}
			}
		}
		else
		{
			Sample::MouseDown( p, button, mod );
		}
	}

	// --- Visualization helpers ---

	static ImU32 ValueColor( float v )
	{
		float t = ( v + 1.0f ) * 0.5f;
		t = t < 0.0f ? 0.0f : ( t > 1.0f ? 1.0f : t );
		int r, g, b;
		if ( t < 0.5f )
		{
			float s = t / 0.5f;
			r = (int)( 60 + s * 195 );
			g = (int)( 60 + s * 195 );
			b = 255;
		}
		else
		{
			float s = ( t - 0.5f ) / 0.5f;
			r = 255;
			g = (int)( 255 - s * 195 );
			b = (int)( 255 - s * 195 );
		}
		return IM_COL32( r, g, b, 255 );
	}

	void DrawNeuralNetWindow( const char* title, const float* inputs, int numInputs, const float* hidden,
							  int numHidden, const float* outputs, int numOutputs, const char** inputLabels,
							  int numInputLabels, const char** outputLabels, int numOutputLabels, const char* agentInfo )
	{
		float fontSize = ImGui::GetFontSize();
		ImGui::SetNextWindowPos( ImVec2( m_camera->width - 28.0f * fontSize, 1.0f * fontSize ), ImGuiCond_Once );
		ImGui::SetNextWindowSize( ImVec2( 27.0f * fontSize, 30.0f * fontSize ), ImGuiCond_Once );
		ImGui::Begin( title, nullptr, ImGuiWindowFlags_NoCollapse );

		ImGui::TextWrapped( "%s", agentInfo );
		ImGui::Separator();

		ImDrawList* drawList = ImGui::GetWindowDrawList();
		ImVec2 canvasPos = ImGui::GetCursorScreenPos();
		float canvasW = ImGui::GetContentRegionAvail().x;
		float canvasH = 18.0f * fontSize;
		ImGui::Dummy( ImVec2( canvasW, canvasH ) );

		float colX[3] = { canvasPos.x + 35.0f, canvasPos.x + canvasW * 0.5f, canvasPos.x + canvasW - 35.0f };
		float nodeR = fontSize * 0.4f;
		float topY = canvasPos.y + nodeR + 2.0f;
		float layerH = canvasH - nodeR * 2 - 4.0f;

		auto ny = [&]( int count, int idx ) -> float
		{
			if ( count <= 1 )
				return topY + layerH * 0.5f;
			return topY + ( layerH * idx ) / ( count - 1 );
		};

		int dispInputs = std::min( numInputLabels, numInputs );

		// Connections input→hidden
		for ( int i = 0; i < dispInputs; ++i )
		{
			ImVec2 from = { colX[0], ny( dispInputs, i ) };
			for ( int h = 0; h < numHidden; ++h )
			{
				ImVec2 to = { colX[1], ny( numHidden, h ) };
				ImU32 col = IM_COL32( 100, 100, 100, 30 + (int)( fabsf( hidden[h] ) * 80 ) );
				drawList->AddLine( from, to, col, 1.0f );
			}
		}

		// Connections hidden→output
		for ( int h = 0; h < numHidden; ++h )
		{
			ImVec2 from = { colX[1], ny( numHidden, h ) };
			for ( int o = 0; o < numOutputs; ++o )
			{
				ImVec2 to = { colX[2], ny( numOutputs, o ) };
				ImU32 col = IM_COL32( 100, 100, 100, 30 + (int)( fabsf( outputs[o] ) * 120 ) );
				drawList->AddLine( from, to, col, 1.0f );
			}
		}

		// Input nodes
		for ( int i = 0; i < dispInputs; ++i )
		{
			ImVec2 c = { colX[0], ny( dispInputs, i ) };
			drawList->AddCircleFilled( c, nodeR, ValueColor( inputs[i] ) );
			drawList->AddCircle( c, nodeR, IM_COL32( 200, 200, 200, 255 ), 0, 1.0f );
			if ( i < numInputLabels )
				drawList->AddText( { c.x - nodeR - fontSize * 3.0f, c.y - fontSize * 0.4f },
								   IM_COL32( 200, 200, 200, 255 ), inputLabels[i] );
		}

		// Hidden nodes
		for ( int h = 0; h < numHidden; ++h )
		{
			ImVec2 c = { colX[1], ny( numHidden, h ) };
			drawList->AddCircleFilled( c, nodeR * 0.7f, ValueColor( hidden[h] ) );
			drawList->AddCircle( c, nodeR * 0.7f, IM_COL32( 200, 200, 200, 255 ), 0, 1.0f );
		}

		// Output nodes
		for ( int o = 0; o < numOutputs; ++o )
		{
			ImVec2 c = { colX[2], ny( numOutputs, o ) };
			drawList->AddCircleFilled( c, nodeR, ValueColor( outputs[o] ) );
			drawList->AddCircle( c, nodeR, IM_COL32( 200, 200, 200, 255 ), 0, 1.0f );
			if ( o < numOutputLabels )
				drawList->AddText( { c.x + nodeR + 4.0f, c.y - fontSize * 0.4f }, IM_COL32( 200, 200, 200, 255 ),
								   outputLabels[o] );
			char buf[16];
			snprintf( buf, sizeof( buf ), "%.2f", outputs[o] );
			drawList->AddText( { c.x - fontSize * 0.5f, c.y - fontSize * 0.4f }, IM_COL32( 0, 0, 0, 255 ), buf );
		}

		// Column labels
		drawList->AddText( { colX[0] - fontSize, canvasPos.y - 2.0f }, IM_COL32( 255, 255, 255, 200 ), "Input" );
		drawList->AddText( { colX[1] - fontSize * 0.8f, canvasPos.y - 2.0f }, IM_COL32( 255, 255, 255, 200 ),
						   "Hidden" );
		drawList->AddText( { colX[2] - fontSize, canvasPos.y - 2.0f }, IM_COL32( 255, 255, 255, 200 ), "Output" );

		ImGui::End();
	}

	void DrawTrainingPlots()
	{
		if ( m_historyCount < 2 )
			return;

		float fontSize = ImGui::GetFontSize();
		ImGui::SetNextWindowPos( ImVec2( m_camera->width - 28.0f * fontSize, 32.0f * fontSize ), ImGuiCond_Once );
		ImGui::SetNextWindowSize( ImVec2( 27.0f * fontSize, 18.0f * fontSize ), ImGuiCond_Once );
		ImGui::Begin( "Training", nullptr, ImGuiWindowFlags_NoCollapse );

		ImVec2 plotSize = { -1, 6.0f * fontSize };

		if ( ImPlot::BeginPlot( "Fitness", plotSize, ImPlotFlags_NoTitle ) )
		{
			ImPlot::SetupAxes( "Gen", "Fitness" );
			ImPlot::SetupAxesLimits( 0, m_historyCount + 1, 0, 0, ImPlotCond_Always );
			ImPlot::SetupAxisLimits( ImAxis_Y1, 0, 0, ImPlotCond_Once );
			ImPlot::PlotLine( "Sharks", m_histGeneration, m_histSharkFitness, m_historyCount );
			ImPlot::PlotLine( "Fish", m_histGeneration, m_histFishFitness, m_historyCount );
			ImPlot::EndPlot();
		}

		if ( ImPlot::BeginPlot( "Population", plotSize, ImPlotFlags_NoTitle ) )
		{
			ImPlot::SetupAxes( "Gen", "Count" );
			ImPlot::SetupAxesLimits( 0, m_historyCount + 1, 0, 0, ImPlotCond_Always );
			ImPlot::SetupAxisLimits( ImAxis_Y1, 0, 0, ImPlotCond_Once );
			ImPlot::PlotLine( "Fish Alive", m_histGeneration, m_histFishAlive, m_historyCount );
			ImPlot::PlotLine( "Fish Caught", m_histGeneration, m_histFishCaught, m_historyCount );
			ImPlot::EndPlot();
		}

		ImGui::End();
	}

	void Step() override
	{
		float dt = 1.0f / m_context->hertz;

		// Update agents
		for ( int i = 0; i < (int)m_sharks.size(); ++i )
			UpdateShark( m_sharks[i], i, dt );
		for ( int i = 0; i < (int)m_fish.size(); ++i )
			UpdateFish( m_fish[i], i, dt );

		Sample::Step();

		ProcessContacts();

		m_generationTimer += dt;
		if ( m_generationTimer >= m_generationDuration )
			Evolve();

		// --- Draw fish (draw first so sharks appear on top) ---
		for ( int i = 0; i < (int)m_fish.size(); ++i )
		{
			Fish& f = m_fish[i];
			if ( !f.alive )
				continue;

			b2Vec2 pos = b2Body_GetPosition( f.bodyId );

			// Draw a small triangle (fish shape) pointing in heading direction
			float r = FISH_RADIUS;
			float ha = f.heading;
			b2Vec2 tip = { pos.x + cosf( ha ) * r * 2.0f, pos.y + sinf( ha ) * r * 2.0f };
			b2Vec2 left = { pos.x + cosf( ha + 2.4f ) * r * 1.2f, pos.y + sinf( ha + 2.4f ) * r * 1.2f };
			b2Vec2 right = { pos.x + cosf( ha - 2.4f ) * r * 1.2f, pos.y + sinf( ha - 2.4f ) * r * 1.2f };

			DrawLine( m_draw, tip, left, COLOR_FISH );
			DrawLine( m_draw, left, right, COLOR_FISH );
			DrawLine( m_draw, right, tip, COLOR_FISH );

			if ( i == m_selectedFish )
				DrawCircle( m_draw, pos, FISH_RADIUS * 3.0f, b2_colorCyan );
		}

		// --- Draw sharks ---
		for ( int i = 0; i < (int)m_sharks.size(); ++i )
		{
			SharkAgent& s = m_sharks[i];
			if ( !s.alive )
				continue;

			b2Vec2 pos = b2Body_GetPosition( s.bodyId );
			float rad = SharkRadius( s.sharkType );
			b2HexColor col = SharkColor( s.sharkType );

			b2Transform xf = { pos, b2MakeRot( s.heading ) };
			DrawSolidCircle( m_draw, xf, rad, col );

			// Heading line
			b2Vec2 headEnd = { pos.x + cosf( s.heading ) * rad * 2.0f, pos.y + sinf( s.heading ) * rad * 2.0f };
			DrawLine( m_draw, pos, headEnd, b2_colorWhite );

			// Shark name above
			if ( i == m_selectedShark )
				DrawCircle( m_draw, pos, rad * 2.5f, b2_colorCyan );
		}

		// HUD
		int aliveFish = 0;
		for ( auto& f : m_fish )
			if ( f.alive )
				aliveFish++;
		int aliveSharks = 0;
		for ( auto& s : m_sharks )
			if ( s.alive )
				aliveSharks++;

		DrawColoredTextLine( COLOR_BABY, "=== Baby Shark Ocean ===" );
		DrawTextLine( "Gen: %d  |  Time: %.1f / %.1f", m_generation, m_generationTimer, m_generationDuration );
		DrawTextLine( "Sharks: %d  |  Fish: %d/%d  |  Caught: %d", aliveSharks, aliveFish, m_fishCount,
					  m_totalFishCaught );
		DrawTextLine( "Shift+Click to select agent" );

		// Neural net window for selected agent
		if ( m_selectedShark >= 0 && m_selectedShark < (int)m_sharks.size() && m_sharks[m_selectedShark].alive )
		{
			SharkAgent& s = m_sharks[m_selectedShark];
			static const char* sharkInputLabels[] = { "Dist", "Fish", "Shark", "Wall", "Enrg", "Spd", "AllyX",
													  "AllyY" };
			static const char* sharkOutputLabels[] = { "Turn", "Speed" };

			// Summarize inputs for net display
			float summary[8];
			for ( int c = 0; c < 4; ++c )
			{
				float mx = 0.0f;
				for ( int r = 0; r < SHARK_RAYS; ++r )
				{
					float v = s.lastInputs[r * SHARK_RAY_CHANNELS + c];
					if ( fabsf( v ) > fabsf( mx ) )
						mx = v;
				}
				summary[c] = mx;
			}
			int eb = SHARK_RAYS * SHARK_RAY_CHANNELS;
			summary[4] = s.lastInputs[eb + 0];
			summary[5] = s.lastInputs[eb + 1];
			summary[6] = s.lastInputs[eb + 2];
			summary[7] = s.lastInputs[eb + 3];

			char info[128];
			snprintf( info, sizeof( info ), "%s Shark #%d | Energy: %.0f | Caught: %d | Fitness: %.0f",
					  SharkName( s.sharkType ), m_selectedShark, s.energy, s.fishCaught, s.fitness );

			DrawNeuralNetWindow( "Shark Brain", summary, 8, s.lastHidden, SHARK_NN_HIDDEN, s.lastOutputs,
								SHARK_NN_OUTPUTS, sharkInputLabels, 8, sharkOutputLabels, 2, info );
		}
		else if ( m_selectedFish >= 0 && m_selectedFish < (int)m_fish.size() && m_fish[m_selectedFish].alive )
		{
			Fish& f = m_fish[m_selectedFish];
			static const char* fishInputLabels[] = { "Dist", "Shark", "Fish", "Wall", "NbrX", "NbrY", "Spd", "Enrg" };
			static const char* fishOutputLabels[] = { "Turn", "Speed" };

			float summary[8];
			for ( int c = 0; c < 4; ++c )
			{
				float mx = 0.0f;
				for ( int r = 0; r < FISH_RAYS; ++r )
				{
					float v = f.lastInputs[r * FISH_RAY_CHANNELS + c];
					if ( fabsf( v ) > fabsf( mx ) )
						mx = v;
				}
				summary[c] = mx;
			}
			int eb = FISH_RAYS * FISH_RAY_CHANNELS;
			summary[4] = f.lastInputs[eb + 0];
			summary[5] = f.lastInputs[eb + 1];
			summary[6] = f.lastInputs[eb + 2];
			summary[7] = f.lastInputs[eb + 3];

			char info[128];
			snprintf( info, sizeof( info ), "Fish #%d | Survival: %.1fs | Energy: %.0f | Fitness: %.0f",
					  m_selectedFish, f.age, f.energy, f.fitness );

			DrawNeuralNetWindow( "Fish Brain", summary, 8, f.lastHidden, FISH_NN_HIDDEN, f.lastOutputs,
								FISH_NN_OUTPUTS, fishInputLabels, 8, fishOutputLabels, 2, info );
		}

		DrawTrainingPlots();
	}

	void UpdateGui() override
	{
		float fontSize = ImGui::GetFontSize();
		ImGui::SetNextWindowPos( ImVec2( 0.5f * fontSize, m_camera->height - 16.0f * fontSize ), ImGuiCond_Once );
		ImGui::SetNextWindowSize( ImVec2( 14.0f * fontSize, 15.0f * fontSize ) );
		ImGui::Begin( "Baby Shark", nullptr, ImGuiWindowFlags_NoResize );

		ImGui::TextColored( ImVec4( 0.3f, 0.76f, 0.97f, 1 ), "Shark Family" );
		ImGui::SliderInt( "Baby", &m_babyCount, 1, 10 );
		ImGui::SliderInt( "Mama", &m_mamaCount, 1, 5 );
		ImGui::SliderInt( "Daddy", &m_daddyCount, 1, 3 );

		ImGui::Separator();
		ImGui::TextColored( ImVec4( 1, 0.84f, 0.31f, 1 ), "Fish School" );
		ImGui::SliderInt( "Fish", &m_fishCount, 10, 150 );

		ImGui::Separator();
		ImGui::SliderFloat( "Gen Duration", &m_generationDuration, 5.0f, 60.0f, "%.0f s" );
		ImGui::SliderFloat( "Mutation", &m_mutationRate, 0.01f, 0.5f );
		ImGui::SliderFloat( "Mut Str", &m_mutationStrength, 0.1f, 2.0f );
		ImGui::Checkbox( "Show Rays", &m_showRays );

		if ( ImGui::Button( "Reset" ) )
		{
			for ( auto& s : m_sharks )
				if ( b2Body_IsValid( s.bodyId ) )
					b2DestroyBody( s.bodyId );
			for ( auto& f : m_fish )
				if ( b2Body_IsValid( f.bodyId ) )
					b2DestroyBody( f.bodyId );
			for ( auto& tag : m_tags )
				delete tag;
			m_tags.clear();
			m_sharks.clear();
			m_fish.clear();
			m_generation = 0;
			m_generationTimer = 0.0f;
			m_historyCount = 0;
			m_selectedShark = -1;
			m_selectedFish = -1;
			m_totalFishCaught = 0;
			SpawnAll();
		}

		if ( ImGui::Button( "Force Evolve" ) )
			Evolve();

		ImGui::End();
	}

	static Sample* Create( SampleContext* context )
	{
		return new BabySharkOcean( context );
	}

	// --- Data ---
	std::vector<SharkAgent> m_sharks;
	std::vector<Fish> m_fish;
	std::vector<EntityTag*> m_tags;

	int m_babyCount;
	int m_mamaCount;
	int m_daddyCount;
	int m_fishCount;
	float m_mutationRate;
	float m_mutationStrength;
	int m_generation;
	float m_generationTimer;
	float m_generationDuration;
	bool m_showRays;
	int m_selectedShark;
	int m_selectedFish;
	int m_totalFishCaught;
	float m_bestSharkFitness = 0.0f;
	float m_bestFishFitness = 0.0f;

	static constexpr int MAX_HISTORY = 500;
	float m_histGeneration[MAX_HISTORY] = {};
	float m_histSharkFitness[MAX_HISTORY] = {};
	float m_histFishFitness[MAX_HISTORY] = {};
	float m_histFishAlive[MAX_HISTORY] = {};
	float m_histFishCaught[MAX_HISTORY] = {};
	int m_historyCount;
};

static int sampleBabyShark = RegisterSample( "AI", "Baby Shark Ocean", BabySharkOcean::Create );
