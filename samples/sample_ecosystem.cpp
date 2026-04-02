// SPDX-FileCopyrightText: 2024 Erin Catto
// SPDX-License-Identifier: MIT

#include "draw.h"
#include "random.h"
#include "sample.h"

#include "box2d/box2d.h"
#include "box2d/math_functions.h"

#include <imgui.h>
#include <implot.h>
#include <GLFW/glfw3.h>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <vector>

// --- Simple feedforward neural network ---

struct NeuralNet
{
	// Architecture: inputSize -> hiddenSize -> outputSize
	// Weights stored flat: [input->hidden weights, hidden biases, hidden->output weights, output biases]
	std::vector<float> weights;
	int inputSize;
	int hiddenSize;
	int outputSize;

	void Init( int inSize, int hSize, int outSize )
	{
		inputSize = inSize;
		hiddenSize = hSize;
		outputSize = outSize;
		int totalWeights = ( inSize * hSize ) + hSize + ( hSize * outSize ) + outSize;
		weights.resize( totalWeights );
		Randomize();
	}

	void Randomize()
	{
		for ( size_t i = 0; i < weights.size(); ++i )
		{
			weights[i] = RandomFloat() * 2.0f; // range [-2, 2]
		}
	}

	void Forward( const float* input, float* output ) const
	{
		int idx = 0;

		// Hidden layer
		std::vector<float> hidden( hiddenSize );
		for ( int h = 0; h < hiddenSize; ++h )
		{
			float sum = 0.0f;
			for ( int i = 0; i < inputSize; ++i )
			{
				sum += input[i] * weights[idx++];
			}
			sum += weights[idx++]; // bias
			hidden[h] = tanhf( sum );
		}

		// Output layer
		for ( int o = 0; o < outputSize; ++o )
		{
			float sum = 0.0f;
			for ( int h = 0; h < hiddenSize; ++h )
			{
				sum += hidden[h] * weights[idx++];
			}
			sum += weights[idx++]; // bias
			output[o] = tanhf( sum );
		}
	}

	void Mutate( float rate, float strength )
	{
		for ( size_t i = 0; i < weights.size(); ++i )
		{
			float r = (float)RandomInt() / (float)RAND_LIMIT;
			if ( r < rate )
			{
				weights[i] += RandomFloat() * strength;
			}
		}
	}

	void CopyFrom( const NeuralNet& other )
	{
		weights = other.weights;
		inputSize = other.inputSize;
		hiddenSize = other.hiddenSize;
		outputSize = other.outputSize;
	}

	// Forward pass that stores all activations for visualization
	void ForwardStore( const float* input, float* output, float* hiddenOut ) const
	{
		int idx = 0;

		// Hidden layer
		for ( int h = 0; h < hiddenSize; ++h )
		{
			float sum = 0.0f;
			for ( int i = 0; i < inputSize; ++i )
			{
				sum += input[i] * weights[idx++];
			}
			sum += weights[idx++]; // bias
			hiddenOut[h] = tanhf( sum );
		}

		// Output layer
		for ( int o = 0; o < outputSize; ++o )
		{
			float sum = 0.0f;
			for ( int h = 0; h < hiddenSize; ++h )
			{
				sum += hiddenOut[h] * weights[idx++];
			}
			sum += weights[idx++]; // bias
			output[o] = tanhf( sum );
		}
	}
};

// --- Agent types ---

enum AgentType
{
	e_prey = 0,
	e_predator = 1,
	e_food = 2
};

struct Agent
{
	b2BodyId bodyId;
	b2ShapeId shapeId;
	AgentType type;
	NeuralNet brain;
	float energy;
	float heading; // radians
	float age;
	bool alive;
	float fitness;

	// Last-frame activations for visualization
	float lastInputs[5 * 12 + 2];   // NN_INPUTS
	float lastHidden[16];            // NN_HIDDEN
	float lastOutputs[2];            // NN_OUTPUTS
};

// Neural net input layout (per agent):
// For each of N_RAYS rays: [distance, isPrey, isPredator, isFood, isWall] => 5 * N_RAYS
// Plus: [own energy normalized, own speed normalized] => 2
// Total inputs: 5 * N_RAYS + 2
// Outputs: [turn amount, speed]

static constexpr int N_RAYS = 12;
static constexpr int NN_INPUTS = 5 * N_RAYS + 2;
static constexpr int NN_HIDDEN = 16;
static constexpr int NN_OUTPUTS = 2;

static constexpr float PREY_RADIUS = 0.3f;
static constexpr float PREDATOR_RADIUS = 0.45f;
static constexpr float FOOD_RADIUS = 0.2f;

static constexpr float PREY_MAX_SPEED = 6.0f;
static constexpr float PREDATOR_MAX_SPEED = 8.0f;

static constexpr float PREY_VISION_RANGE = 12.0f;
static constexpr float PREDATOR_VISION_RANGE = 15.0f;

static constexpr float ARENA_HALF_SIZE = 30.0f;

// Category bits for filtering
static constexpr uint64_t CAT_WALL = 0x0001;
static constexpr uint64_t CAT_PREY = 0x0002;
static constexpr uint64_t CAT_PREDATOR = 0x0004;
static constexpr uint64_t CAT_FOOD = 0x0008;

class PreyPredator : public Sample
{
public:
	explicit PreyPredator( SampleContext* context )
		: Sample( context )
	{
		if ( m_context->restart == false )
		{
			m_context->camera.center = { 0.0f, 0.0f };
			m_context->camera.zoom = 25.0f * 2.5f;
		}

		m_preyCount = 30;
		m_predatorCount = 8;
		m_foodCount = 40;
		m_mutationRate = 0.15f;
		m_mutationStrength = 0.5f;
		m_generation = 0;
		m_generationTimer = 0.0f;
		m_generationDuration = 30.0f; // seconds per generation
		m_showRays = false;
		m_fastForward = false;
		m_preyEaten = 0;
		m_preyStarved = 0;
		m_selectedAgent = -1;

		CreateArena();
		SpawnAgents();
	}

	void CreateArena()
	{
		b2BodyDef bodyDef = b2DefaultBodyDef();
		b2BodyId groundId = b2CreateBody( m_worldId, &bodyDef );

		b2ShapeDef shapeDef = b2DefaultShapeDef();
		shapeDef.filter.categoryBits = CAT_WALL;
		shapeDef.filter.maskBits = CAT_PREY | CAT_PREDATOR;

		float hs = ARENA_HALF_SIZE;
		b2Segment segments[4] = {
			{ { -hs, -hs }, { hs, -hs } },
			{ { hs, -hs }, { hs, hs } },
			{ { hs, hs }, { -hs, hs } },
			{ { -hs, hs }, { -hs, -hs } },
		};

		for ( int i = 0; i < 4; ++i )
		{
			b2CreateSegmentShape( groundId, &shapeDef, &segments[i] );
		}
	}

	b2ShapeId CreateAgentBody( b2Vec2 position, float radius, uint64_t category, uint64_t mask, void* userData,
							   b2BodyId* outBodyId )
	{
		b2BodyDef bodyDef = b2DefaultBodyDef();
		bodyDef.type = b2_dynamicBody;
		bodyDef.position = position;
		bodyDef.linearDamping = 3.0f;
		bodyDef.motionLocks.angularZ = true;
		bodyDef.userData = userData;

		b2BodyId bodyId = b2CreateBody( m_worldId, &bodyDef );
		*outBodyId = bodyId;

		b2ShapeDef shapeDef = b2DefaultShapeDef();
		shapeDef.density = 1.0f;
		shapeDef.filter.categoryBits = category;
		shapeDef.filter.maskBits = mask;
		shapeDef.enableContactEvents = true;
		shapeDef.enableSensorEvents = true;

		b2Circle circle = { { 0.0f, 0.0f }, radius };
		b2ShapeId shapeId = b2CreateCircleShape( bodyId, &shapeDef, &circle );

		return shapeId;
	}

	void SpawnAgents()
	{
		m_agents.clear();
		m_foods.clear();

		float hs = ARENA_HALF_SIZE - 2.0f;

		// Reserve to prevent reallocation (userData pointers must stay valid)
		m_agents.reserve( m_preyCount + m_predatorCount );

		// Create agent data first
		for ( int i = 0; i < m_preyCount; ++i )
		{
			Agent agent = {};
			agent.type = e_prey;
			agent.energy = 100.0f;
			agent.heading = RandomFloatRange( -B2_PI, B2_PI );
			agent.age = 0.0f;
			agent.alive = true;
			agent.fitness = 0.0f;
			agent.brain.Init( NN_INPUTS, NN_HIDDEN, NN_OUTPUTS );
			m_agents.push_back( agent );
		}

		for ( int i = 0; i < m_predatorCount; ++i )
		{
			Agent agent = {};
			agent.type = e_predator;
			agent.energy = 150.0f;
			agent.heading = RandomFloatRange( -B2_PI, B2_PI );
			agent.age = 0.0f;
			agent.alive = true;
			agent.fitness = 0.0f;
			agent.brain.Init( NN_INPUTS, NN_HIDDEN, NN_OUTPUTS );
			m_agents.push_back( agent );
		}

		// Now create bodies (pointers are stable since vector won't reallocate)
		for ( int i = 0; i < (int)m_agents.size(); ++i )
		{
			Agent& a = m_agents[i];
			b2Vec2 pos = { RandomFloatRange( -hs, hs ), RandomFloatRange( -hs, hs ) };
			float radius = ( a.type == e_predator ) ? PREDATOR_RADIUS : PREY_RADIUS;
			uint64_t cat = ( a.type == e_predator ) ? CAT_PREDATOR : CAT_PREY;
			uint64_t mask = ( a.type == e_predator ) ? ( CAT_WALL | CAT_PREY ) : ( CAT_WALL | CAT_PREDATOR | CAT_FOOD );
			a.shapeId = CreateAgentBody( pos, radius, cat, mask, &a, &a.bodyId );
		}

		// Spawn food
		SpawnFood( m_foodCount );
	}

	void SpawnFood( int count )
	{
		float hs = ARENA_HALF_SIZE - 2.0f;
		for ( int i = 0; i < count; ++i )
		{
			Agent food = {};
			food.type = e_food;
			food.alive = true;
			food.energy = 30.0f;

			b2Vec2 pos = { RandomFloatRange( -hs, hs ), RandomFloatRange( -hs, hs ) };

			b2BodyDef bodyDef = b2DefaultBodyDef();
			bodyDef.type = b2_staticBody;
			bodyDef.position = pos;
			food.bodyId = b2CreateBody( m_worldId, &bodyDef );

			b2ShapeDef shapeDef = b2DefaultShapeDef();
			shapeDef.filter.categoryBits = CAT_FOOD;
			shapeDef.filter.maskBits = CAT_PREY;
			shapeDef.isSensor = true;
			shapeDef.enableSensorEvents = true;

			b2Circle circle = { { 0.0f, 0.0f }, FOOD_RADIUS };
			food.shapeId = b2CreateCircleShape( food.bodyId, &shapeDef, &circle );

			m_foods.push_back( food );
		}
	}

	void CastRaysForAgent( Agent& agent, float* inputs, bool forceDrawRays = false )
	{
		b2Vec2 pos = b2Body_GetPosition( agent.bodyId );
		float visionRange = ( agent.type == e_predator ) ? PREDATOR_VISION_RANGE : PREY_VISION_RANGE;

		b2QueryFilter filter = b2DefaultQueryFilter();
		filter.categoryBits = CAT_PREY | CAT_PREDATOR | CAT_FOOD | CAT_WALL;
		filter.maskBits = CAT_PREY | CAT_PREDATOR | CAT_FOOD | CAT_WALL;

		float angleStep = 2.0f * B2_PI / (float)N_RAYS;

		for ( int r = 0; r < N_RAYS; ++r )
		{
			float angle = agent.heading + ( r - N_RAYS / 2 ) * angleStep;
			b2Vec2 dir = { cosf( angle ), sinf( angle ) };
			b2Vec2 translation = { dir.x * visionRange, dir.y * visionRange };

			b2RayResult result = b2World_CastRayClosest( m_worldId, pos, translation, filter );

			int base = r * 5;
			if ( result.hit )
			{
				inputs[base + 0] = 1.0f - result.fraction; // closer = higher

				// Determine what was hit
				b2BodyId hitBody = b2Shape_GetBody( result.shapeId );
				void* ud = b2Body_GetUserData( hitBody );
				inputs[base + 1] = 0.0f; // isPrey
				inputs[base + 2] = 0.0f; // isPredator
				inputs[base + 3] = 0.0f; // isFood
				inputs[base + 4] = 0.0f; // isWall

				if ( ud != nullptr )
				{
					Agent* hitAgent = (Agent*)ud;
					if ( hitAgent->type == e_prey )
						inputs[base + 1] = 1.0f;
					else if ( hitAgent->type == e_predator )
						inputs[base + 2] = 1.0f;
					else if ( hitAgent->type == e_food )
						inputs[base + 3] = 1.0f;
				}
				else
				{
					inputs[base + 4] = 1.0f; // wall
				}

				// Draw ray for visualization
				if ( m_showRays || forceDrawRays )
				{
					b2Vec2 hitPoint = { pos.x + translation.x * result.fraction, pos.y + translation.y * result.fraction };
					b2HexColor color = b2_colorGray;
					if ( inputs[base + 1] > 0.5f )
						color = b2_colorGreen;
					else if ( inputs[base + 2] > 0.5f )
						color = b2_colorRed;
					else if ( inputs[base + 3] > 0.5f )
						color = b2_colorYellow;
					DrawLine( m_draw, pos, hitPoint, color );
				}
			}
			else
			{
				inputs[base + 0] = 0.0f;
				inputs[base + 1] = 0.0f;
				inputs[base + 2] = 0.0f;
				inputs[base + 3] = 0.0f;
				inputs[base + 4] = 0.0f;
			}
		}

		// Extra inputs
		float maxEnergy = ( agent.type == e_predator ) ? 200.0f : 150.0f;
		inputs[N_RAYS * 5 + 0] = agent.energy / maxEnergy;

		b2Vec2 vel = b2Body_GetLinearVelocity( agent.bodyId );
		float speed = b2Length( vel );
		float maxSpeed = ( agent.type == e_predator ) ? PREDATOR_MAX_SPEED : PREY_MAX_SPEED;
		inputs[N_RAYS * 5 + 1] = speed / maxSpeed;
	}

	void UpdateAgent( Agent& agent, float timeStep )
	{
		if ( !agent.alive )
			return;

		// Gather inputs
		float inputs[NN_INPUTS];
		memset( inputs, 0, sizeof( inputs ) );
		CastRaysForAgent( agent, inputs );

		// Run neural network and store activations
		float outputs[NN_OUTPUTS];
		float hidden[NN_HIDDEN];
		agent.brain.ForwardStore( inputs, outputs, hidden );

		// Store for visualization
		memcpy( agent.lastInputs, inputs, sizeof( inputs ) );
		memcpy( agent.lastHidden, hidden, sizeof( hidden ) );
		memcpy( agent.lastOutputs, outputs, sizeof( outputs ) );

		// outputs[0] = turn amount [-1, 1] mapped to [-PI/4, PI/4] per step
		// outputs[1] = speed [0, 1] (tanh output mapped)
		agent.heading += outputs[0] * ( B2_PI / 4.0f ) * timeStep;

		float maxSpeed = ( agent.type == e_predator ) ? PREDATOR_MAX_SPEED : PREY_MAX_SPEED;
		float speed = ( outputs[1] + 1.0f ) * 0.5f * maxSpeed; // map [-1,1] to [0, maxSpeed]

		b2Vec2 velocity = { cosf( agent.heading ) * speed, sinf( agent.heading ) * speed };
		b2Body_SetLinearVelocity( agent.bodyId, velocity );

		// Energy cost: moving costs more
		float moveCost = 0.5f + speed * 0.3f;
		agent.energy -= moveCost * timeStep;
		agent.age += timeStep;

		// Fitness accumulation
		if ( agent.type == e_prey )
		{
			agent.fitness += timeStep; // reward survival time
		}

		// Death by starvation
		if ( agent.energy <= 0.0f )
		{
			agent.alive = false;
			if ( agent.type == e_prey )
				m_preyStarved++;
			b2Body_Disable( agent.bodyId );
		}
	}

	void ProcessContacts()
	{
		// Sensor events for prey eating food
		b2SensorEvents sensorEvents = b2World_GetSensorEvents( m_worldId );
		for ( int i = 0; i < sensorEvents.beginCount; ++i )
		{
			b2SensorBeginTouchEvent event = sensorEvents.beginEvents[i];
			// The sensor is the food, the visitor is the prey
			b2BodyId sensorBody = b2Shape_GetBody( event.sensorShapeId );
			b2BodyId visitorBody = b2Shape_GetBody( event.visitorShapeId );

			// Find food and mark eaten
			for ( auto& food : m_foods )
			{
				if ( food.alive && B2_ID_EQUALS( food.bodyId, sensorBody ) )
				{
					// Find the prey
					for ( auto& agent : m_agents )
					{
						if ( agent.alive && agent.type == e_prey && B2_ID_EQUALS( agent.bodyId, visitorBody ) )
						{
							agent.energy += food.energy;
							agent.fitness += 20.0f; // bonus for eating
							food.alive = false;
							b2Body_Disable( food.bodyId );
							break;
						}
					}
					break;
				}
			}
		}

		// Contact events for predator catching prey
		b2ContactEvents contactEvents = b2World_GetContactEvents( m_worldId );
		for ( int i = 0; i < contactEvents.beginCount; ++i )
		{
			b2ContactBeginTouchEvent event = contactEvents.beginEvents[i];
			b2BodyId bodyA = b2Shape_GetBody( event.shapeIdA );
			b2BodyId bodyB = b2Shape_GetBody( event.shapeIdB );

			Agent* predator = nullptr;
			Agent* prey = nullptr;

			void* udA = b2Body_GetUserData( bodyA );
			void* udB = b2Body_GetUserData( bodyB );

			if ( udA && udB )
			{
				Agent* agentA = (Agent*)udA;
				Agent* agentB = (Agent*)udB;

				if ( agentA->type == e_predator && agentB->type == e_prey )
				{
					predator = agentA;
					prey = agentB;
				}
				else if ( agentB->type == e_predator && agentA->type == e_prey )
				{
					predator = agentB;
					prey = agentA;
				}
			}

			if ( predator && prey && predator->alive && prey->alive )
			{
				prey->alive = false;
				b2Body_Disable( prey->bodyId );
				predator->energy += 60.0f;
				predator->fitness += 50.0f; // big reward for catching prey
				m_preyEaten++;
			}
		}
	}

	void Evolve()
	{
		m_generation++;

		// Separate prey and predator brains
		std::vector<Agent*> preyAgents;
		std::vector<Agent*> predatorAgents;

		for ( auto& a : m_agents )
		{
			if ( a.type == e_prey )
				preyAgents.push_back( &a );
			else if ( a.type == e_predator )
				predatorAgents.push_back( &a );
		}

		// Sort by fitness (descending)
		auto cmp = []( const Agent* a, const Agent* b ) { return a->fitness > b->fitness; };
		std::sort( preyAgents.begin(), preyAgents.end(), cmp );
		std::sort( predatorAgents.begin(), predatorAgents.end(), cmp );

		// Store best brains
		std::vector<NeuralNet> bestPreyBrains;
		std::vector<NeuralNet> bestPredBrains;

		int preyElite = (int)preyAgents.size() > 0 ? ( (int)preyAgents.size() + 3 ) / 4 : 0;
		int predElite = (int)predatorAgents.size() > 0 ? ( (int)predatorAgents.size() + 3 ) / 4 : 0;

		for ( int i = 0; i < preyElite; ++i )
			bestPreyBrains.push_back( preyAgents[i]->brain );
		for ( int i = 0; i < predElite; ++i )
			bestPredBrains.push_back( predatorAgents[i]->brain );

		m_bestPreyFitness = preyElite > 0 ? preyAgents[0]->fitness : 0.0f;
		m_bestPredFitness = predElite > 0 ? predatorAgents[0]->fitness : 0.0f;

		// Record history for plots
		if ( m_historyCount < MAX_HISTORY )
		{
			m_histGeneration[m_historyCount] = (float)m_generation;
			m_histPreyFitness[m_historyCount] = m_bestPreyFitness;
			m_histPredFitness[m_historyCount] = m_bestPredFitness;

			// Count alive at end of generation
			int preyAlive = 0, predAlive = 0;
			for ( auto& a : m_agents )
			{
				if ( a.alive && a.type == e_prey )
					preyAlive++;
				if ( a.alive && a.type == e_predator )
					predAlive++;
			}
			m_histPreySurvival[m_historyCount] = (float)preyAlive;
			m_histPredSurvival[m_historyCount] = (float)predAlive;
			m_histPreyEaten[m_historyCount] = (float)m_preyEaten;
			m_historyCount++;
		}

		// Destroy all current agents and food
		for ( auto& a : m_agents )
		{
			if ( b2Body_IsValid( a.bodyId ) )
				b2DestroyBody( a.bodyId );
		}
		for ( auto& f : m_foods )
		{
			if ( b2Body_IsValid( f.bodyId ) )
				b2DestroyBody( f.bodyId );
		}

		m_agents.clear();
		m_foods.clear();

		// Respawn with evolved brains
		float hs = ARENA_HALF_SIZE - 2.0f;

		// Reserve to prevent reallocation
		m_agents.reserve( m_preyCount + m_predatorCount );

		for ( int i = 0; i < m_preyCount; ++i )
		{
			Agent agent = {};
			agent.type = e_prey;
			agent.energy = 100.0f;
			agent.heading = RandomFloatRange( -B2_PI, B2_PI );
			agent.age = 0.0f;
			agent.alive = true;
			agent.fitness = 0.0f;

			if ( bestPreyBrains.size() > 0 )
			{
				agent.brain.CopyFrom( bestPreyBrains[i % bestPreyBrains.size()] );
				agent.brain.Mutate( m_mutationRate, m_mutationStrength );
			}
			else
			{
				agent.brain.Init( NN_INPUTS, NN_HIDDEN, NN_OUTPUTS );
			}

			m_agents.push_back( agent );
		}

		for ( int i = 0; i < m_predatorCount; ++i )
		{
			Agent agent = {};
			agent.type = e_predator;
			agent.energy = 150.0f;
			agent.heading = RandomFloatRange( -B2_PI, B2_PI );
			agent.age = 0.0f;
			agent.alive = true;
			agent.fitness = 0.0f;

			if ( bestPredBrains.size() > 0 )
			{
				agent.brain.CopyFrom( bestPredBrains[i % bestPredBrains.size()] );
				agent.brain.Mutate( m_mutationRate, m_mutationStrength );
			}
			else
			{
				agent.brain.Init( NN_INPUTS, NN_HIDDEN, NN_OUTPUTS );
			}

			m_agents.push_back( agent );
		}

		// Create bodies after all agents are in the vector (stable pointers)
		for ( int i = 0; i < (int)m_agents.size(); ++i )
		{
			Agent& a = m_agents[i];
			b2Vec2 pos = { RandomFloatRange( -hs, hs ), RandomFloatRange( -hs, hs ) };
			float radius = ( a.type == e_predator ) ? PREDATOR_RADIUS : PREY_RADIUS;
			uint64_t cat = ( a.type == e_predator ) ? CAT_PREDATOR : CAT_PREY;
			uint64_t mask = ( a.type == e_predator ) ? ( CAT_WALL | CAT_PREY ) : ( CAT_WALL | CAT_PREDATOR | CAT_FOOD );
			a.shapeId = CreateAgentBody( pos, radius, cat, mask, &a, &a.bodyId );
		}

		SpawnFood( m_foodCount );

		m_generationTimer = 0.0f;
		m_preyEaten = 0;
		m_preyStarved = 0;
	}

	void RespawnFood()
	{
		// Respawn eaten food periodically
		int aliveFood = 0;
		for ( auto& f : m_foods )
		{
			if ( f.alive )
				aliveFood++;
		}

		if ( aliveFood < m_foodCount / 2 )
		{
			int toSpawn = m_foodCount - aliveFood;
			SpawnFood( toSpawn );
		}
	}

	void MouseDown( b2Vec2 p, int button, int mod ) override
	{
		if ( button == GLFW_MOUSE_BUTTON_1 && ( mod & GLFW_MOD_SHIFT ) )
		{
			// Shift+click to select agent
			float bestDist = 2.0f; // selection radius
			int bestIdx = -1;
			for ( int i = 0; i < (int)m_agents.size(); ++i )
			{
				if ( !m_agents[i].alive )
					continue;
				b2Vec2 apos = b2Body_GetPosition( m_agents[i].bodyId );
				float dx = apos.x - p.x;
				float dy = apos.y - p.y;
				float dist = sqrtf( dx * dx + dy * dy );
				if ( dist < bestDist )
				{
					bestDist = dist;
					bestIdx = i;
				}
			}
			m_selectedAgent = bestIdx;
		}
		else
		{
			Sample::MouseDown( p, button, mod );
		}
	}

	// Helper: map value [-1,1] to color (blue=negative, white=zero, red=positive)
	static ImU32 ValueColor( float v )
	{
		float t = ( v + 1.0f ) * 0.5f; // [0,1]
		if ( t < 0.0f )
			t = 0.0f;
		if ( t > 1.0f )
			t = 1.0f;
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

	void DrawNeuralNetWindow()
	{
		if ( m_selectedAgent < 0 || m_selectedAgent >= (int)m_agents.size() )
			return;

		Agent& agent = m_agents[m_selectedAgent];
		if ( !agent.alive )
		{
			m_selectedAgent = -1;
			return;
		}

		float fontSize = ImGui::GetFontSize();
		ImGui::SetNextWindowPos( ImVec2( m_camera->width - 28.0f * fontSize, 1.0f * fontSize ), ImGuiCond_Once );
		ImGui::SetNextWindowSize( ImVec2( 27.0f * fontSize, 32.0f * fontSize ), ImGuiCond_Once );
		ImGui::Begin( "Neural Network", nullptr, ImGuiWindowFlags_NoCollapse );

		// Agent info header
		const char* typeName = agent.type == e_predator ? "Predator" : "Prey";
		ImGui::TextColored( agent.type == e_predator ? ImVec4( 1, 0.3f, 0.3f, 1 ) : ImVec4( 0.3f, 1, 0.3f, 1 ),
							"%s #%d", typeName, m_selectedAgent );
		ImGui::SameLine();
		ImGui::Text( "Energy: %.0f  Fitness: %.0f  Age: %.1f", agent.energy, agent.fitness, agent.age );

		ImGui::Separator();

		// --- Draw the neural network graph ---
		ImDrawList* drawList = ImGui::GetWindowDrawList();
		ImVec2 canvasPos = ImGui::GetCursorScreenPos();
		float canvasW = ImGui::GetContentRegionAvail().x;
		float canvasH = 20.0f * fontSize;

		// Reserve space
		ImGui::Dummy( ImVec2( canvasW, canvasH ) );

		// Layout: 3 columns (input, hidden, output)
		float colX[3] = { canvasPos.x + 30.0f, canvasPos.x + canvasW * 0.5f, canvasPos.x + canvasW - 30.0f };

		// We only show a subset of inputs (grouped by ray + 2 extras)
		// Show: ray summary (max activation per ray type) + energy + speed = 7 input nodes for readability
		float inputSummary[7];
		const char* inputLabels[7] = { "Dist", "Prey", "Pred", "Food", "Wall", "Enrg", "Spd" };

		// Aggregate ray inputs: average across all rays for each channel
		for ( int c = 0; c < 5; ++c )
		{
			float maxVal = 0.0f;
			for ( int r = 0; r < N_RAYS; ++r )
			{
				float v = agent.lastInputs[r * 5 + c];
				if ( fabsf( v ) > fabsf( maxVal ) )
					maxVal = v;
			}
			inputSummary[c] = maxVal;
		}
		inputSummary[5] = agent.lastInputs[N_RAYS * 5 + 0]; // energy
		inputSummary[6] = agent.lastInputs[N_RAYS * 5 + 1]; // speed

		int numInputNodes = 7;
		int numHiddenNodes = NN_HIDDEN;
		int numOutputNodes = NN_OUTPUTS;

		float nodeRadius = fontSize * 0.45f;

		// Compute node positions
		auto nodeY = [&]( int count, int idx, float topY, float height ) -> float
		{
			if ( count <= 1 )
				return topY + height * 0.5f;
			return topY + ( height * idx ) / ( count - 1 );
		};

		float topY = canvasPos.y + nodeRadius + 2.0f;
		float botY = canvasPos.y + canvasH - nodeRadius - 2.0f;
		float layerH = botY - topY;

		// Draw connections: input -> hidden
		for ( int i = 0; i < numInputNodes; ++i )
		{
			ImVec2 from = { colX[0], nodeY( numInputNodes, i, topY, layerH ) };
			for ( int h = 0; h < numHiddenNodes; ++h )
			{
				ImVec2 to = { colX[1], nodeY( numHiddenNodes, h, topY, layerH ) };
				float w = agent.lastHidden[h]; // connection "importance" approximation
				ImU32 col = IM_COL32( 100, 100, 100, 40 + (int)( fabsf( w ) * 100 ) );
				drawList->AddLine( from, to, col, 1.0f );
			}
		}

		// Draw connections: hidden -> output
		for ( int h = 0; h < numHiddenNodes; ++h )
		{
			ImVec2 from = { colX[1], nodeY( numHiddenNodes, h, topY, layerH ) };
			for ( int o = 0; o < numOutputNodes; ++o )
			{
				ImVec2 to = { colX[2], nodeY( numOutputNodes, o, topY, layerH ) };
				float w = agent.lastOutputs[o];
				ImU32 col = IM_COL32( 100, 100, 100, 40 + (int)( fabsf( w ) * 150 ) );
				drawList->AddLine( from, to, col, 1.0f );
			}
		}

		// Draw input nodes
		for ( int i = 0; i < numInputNodes; ++i )
		{
			ImVec2 center = { colX[0], nodeY( numInputNodes, i, topY, layerH ) };
			ImU32 col = ValueColor( inputSummary[i] );
			drawList->AddCircleFilled( center, nodeRadius, col );
			drawList->AddCircle( center, nodeRadius, IM_COL32( 200, 200, 200, 255 ), 0, 1.0f );

			// Label
			ImVec2 labelPos = { center.x - nodeRadius - fontSize * 3.0f, center.y - fontSize * 0.4f };
			drawList->AddText( labelPos, IM_COL32( 200, 200, 200, 255 ), inputLabels[i] );

			// Value
			char buf[16];
			snprintf( buf, sizeof( buf ), "%.1f", inputSummary[i] );
			drawList->AddText( { center.x - fontSize * 0.5f, center.y - fontSize * 0.4f }, IM_COL32( 0, 0, 0, 255 ),
							   buf );
		}

		// Draw hidden nodes
		for ( int h = 0; h < numHiddenNodes; ++h )
		{
			ImVec2 center = { colX[1], nodeY( numHiddenNodes, h, topY, layerH ) };
			ImU32 col = ValueColor( agent.lastHidden[h] );
			drawList->AddCircleFilled( center, nodeRadius * 0.8f, col );
			drawList->AddCircle( center, nodeRadius * 0.8f, IM_COL32( 200, 200, 200, 255 ), 0, 1.0f );
		}

		// Draw output nodes
		const char* outputLabels[2] = { "Turn", "Speed" };
		for ( int o = 0; o < numOutputNodes; ++o )
		{
			ImVec2 center = { colX[2], nodeY( numOutputNodes, o, topY, layerH ) };
			ImU32 col = ValueColor( agent.lastOutputs[o] );
			drawList->AddCircleFilled( center, nodeRadius, col );
			drawList->AddCircle( center, nodeRadius, IM_COL32( 200, 200, 200, 255 ), 0, 1.0f );

			// Label
			ImVec2 labelPos = { center.x + nodeRadius + 4.0f, center.y - fontSize * 0.4f };
			drawList->AddText( labelPos, IM_COL32( 200, 200, 200, 255 ), outputLabels[o] );

			// Value
			char buf[16];
			snprintf( buf, sizeof( buf ), "%.2f", agent.lastOutputs[o] );
			drawList->AddText( { center.x - fontSize * 0.6f, center.y - fontSize * 0.4f }, IM_COL32( 0, 0, 0, 255 ),
							   buf );
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
		ImGui::SetNextWindowPos( ImVec2( m_camera->width - 28.0f * fontSize, 34.0f * fontSize ), ImGuiCond_Once );
		ImGui::SetNextWindowSize( ImVec2( 27.0f * fontSize, 18.0f * fontSize ), ImGuiCond_Once );
		ImGui::Begin( "Training", nullptr, ImGuiWindowFlags_NoCollapse );

		ImVec2 plotSize = { -1, 6.0f * fontSize };

		// Fitness plot
		if ( ImPlot::BeginPlot( "Best Fitness", plotSize, ImPlotFlags_NoTitle ) )
		{
			ImPlot::SetupAxes( "Gen", "Fitness" );
			ImPlot::SetupAxesLimits( 0, m_historyCount + 1, 0, 0, ImPlotCond_Always );
			ImPlot::SetupAxisLimits( ImAxis_Y1, 0, 0, ImPlotCond_Once );
			ImPlot::PlotLine( "Prey", m_histGeneration, m_histPreyFitness, m_historyCount );
			ImPlot::PlotLine( "Predator", m_histGeneration, m_histPredFitness, m_historyCount );
			ImPlot::EndPlot();
		}

		// Survival/eaten plot
		if ( ImPlot::BeginPlot( "Population", plotSize, ImPlotFlags_NoTitle ) )
		{
			ImPlot::SetupAxes( "Gen", "Count" );
			ImPlot::SetupAxesLimits( 0, m_historyCount + 1, 0, 0, ImPlotCond_Always );
			ImPlot::SetupAxisLimits( ImAxis_Y1, 0, 0, ImPlotCond_Once );
			ImPlot::PlotLine( "Prey Alive", m_histGeneration, m_histPreySurvival, m_historyCount );
			ImPlot::PlotLine( "Pred Alive", m_histGeneration, m_histPredSurvival, m_historyCount );
			ImPlot::PlotLine( "Prey Eaten", m_histGeneration, m_histPreyEaten, m_historyCount );
			ImPlot::EndPlot();
		}

		ImGui::End();
	}

	void Step() override
	{
		float timeStep = 1.0f / m_context->hertz;

		// Update agents before physics step
		for ( auto& agent : m_agents )
		{
			UpdateAgent( agent, timeStep );
		}

		// Physics step
		Sample::Step();

		// Process collisions
		ProcessContacts();

		// Respawn food
		RespawnFood();

		// Generation timer
		m_generationTimer += timeStep;
		if ( m_generationTimer >= m_generationDuration )
		{
			Evolve();
		}

		// Draw agents
		for ( int i = 0; i < (int)m_agents.size(); ++i )
		{
			Agent& agent = m_agents[i];
			if ( !agent.alive )
				continue;

			b2Vec2 pos = b2Body_GetPosition( agent.bodyId );
			float radius = ( agent.type == e_predator ) ? PREDATOR_RADIUS : PREY_RADIUS;
			b2HexColor color = ( agent.type == e_predator ) ? b2_colorRed : b2_colorGreen;

			// Energy-based opacity (dimmer when low energy)
			float maxEnergy = ( agent.type == e_predator ) ? 200.0f : 150.0f;
			float energyFrac = agent.energy / maxEnergy;
			if ( energyFrac < 0.3f )
				color = b2_colorOrange; // warning color

			b2Transform xf = { pos, b2MakeRot( agent.heading ) };
			DrawSolidCircle( m_draw, xf, radius, color );

			// Draw heading line
			b2Vec2 headEnd = { pos.x + cosf( agent.heading ) * radius * 2.0f,
							   pos.y + sinf( agent.heading ) * radius * 2.0f };
			DrawLine( m_draw, pos, headEnd, b2_colorWhite );

			// Highlight selected agent
			if ( i == m_selectedAgent )
			{
				DrawCircle( m_draw, pos, radius * 2.5f, b2_colorCyan );
				// Always show rays for selected agent
				float tmpInputs[NN_INPUTS];
				CastRaysForAgent( agent, tmpInputs, true );
			}
		}

		// Draw food
		for ( auto& food : m_foods )
		{
			if ( !food.alive )
				continue;
			b2Vec2 pos = b2Body_GetPosition( food.bodyId );
			b2Transform xf = { pos, b2MakeRot( 0.0f ) };
			DrawSolidCircle( m_draw, xf, FOOD_RADIUS, b2_colorYellow );
		}

		// HUD
		int alivePreyCount = 0;
		int alivePredCount = 0;
		for ( auto& a : m_agents )
		{
			if ( a.alive && a.type == e_prey )
				alivePreyCount++;
			if ( a.alive && a.type == e_predator )
				alivePredCount++;
		}

		DrawTextLine( "Generation: %d  |  Time: %.1f / %.1f", m_generation, m_generationTimer, m_generationDuration );
		DrawTextLine( "Prey: %d/%d  |  Predators: %d/%d", alivePreyCount, m_preyCount, alivePredCount, m_predatorCount );
		DrawTextLine( "Prey eaten: %d  |  Prey starved: %d", m_preyEaten, m_preyStarved );
		if ( m_selectedAgent >= 0 )
			DrawColoredTextLine( b2_colorCyan, "Shift+Click to select agent | Selected: #%d", m_selectedAgent );
		else
			DrawTextLine( "Shift+Click to select an agent" );

		// Neural net and training windows
		DrawNeuralNetWindow();
		DrawTrainingPlots();
	}

	void UpdateGui() override
	{
		float fontSize = ImGui::GetFontSize();
		ImGui::SetNextWindowPos( ImVec2( 0.5f * fontSize, m_camera->height - 14.0f * fontSize ), ImGuiCond_Once );
		ImGui::SetNextWindowSize( ImVec2( 14.0f * fontSize, 13.0f * fontSize ) );
		ImGui::Begin( "Ecosystem", nullptr, ImGuiWindowFlags_NoResize );

		ImGui::SliderInt( "Prey", &m_preyCount, 5, 100 );
		ImGui::SliderInt( "Predators", &m_predatorCount, 2, 30 );
		ImGui::SliderInt( "Food", &m_foodCount, 10, 100 );
		ImGui::SliderFloat( "Gen Duration", &m_generationDuration, 5.0f, 120.0f, "%.0f s" );
		ImGui::SliderFloat( "Mutation Rate", &m_mutationRate, 0.01f, 0.5f );
		ImGui::SliderFloat( "Mutation Str", &m_mutationStrength, 0.1f, 2.0f );
		ImGui::Checkbox( "Show Rays", &m_showRays );

		if ( ImGui::Button( "Reset" ) )
		{
			// Destroy everything and restart
			for ( auto& a : m_agents )
			{
				if ( b2Body_IsValid( a.bodyId ) )
					b2DestroyBody( a.bodyId );
			}
			for ( auto& f : m_foods )
			{
				if ( b2Body_IsValid( f.bodyId ) )
					b2DestroyBody( f.bodyId );
			}
			m_agents.clear();
			m_foods.clear();
			m_generation = 0;
			m_generationTimer = 0.0f;
			m_bestPreyFitness = 0.0f;
			m_bestPredFitness = 0.0f;
			m_selectedAgent = -1;
			m_historyCount = 0;
			SpawnAgents();
		}

		if ( ImGui::Button( "Force Evolve" ) )
		{
			Evolve();
		}

		ImGui::End();
	}

	static Sample* Create( SampleContext* context )
	{
		return new PreyPredator( context );
	}

	static constexpr int MAX_HISTORY = 500;

	std::vector<Agent> m_agents;
	std::vector<Agent> m_foods;
	int m_preyCount;
	int m_predatorCount;
	int m_foodCount;
	float m_mutationRate;
	float m_mutationStrength;
	int m_generation;
	float m_generationTimer;
	float m_generationDuration;
	bool m_showRays;
	bool m_fastForward;
	int m_preyEaten;
	int m_preyStarved;
	float m_bestPreyFitness = 0.0f;
	float m_bestPredFitness = 0.0f;
	int m_selectedAgent;

	// Training history for plots
	float m_histGeneration[MAX_HISTORY] = {};
	float m_histPreyFitness[MAX_HISTORY] = {};
	float m_histPredFitness[MAX_HISTORY] = {};
	float m_histPreySurvival[MAX_HISTORY] = {};
	float m_histPredSurvival[MAX_HISTORY] = {};
	float m_histPreyEaten[MAX_HISTORY] = {};
	int m_historyCount = 0;
};

static int samplePreyPredator = RegisterSample( "AI", "Prey Predator", PreyPredator::Create );
